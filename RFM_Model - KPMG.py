#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime


# In[2]:


df = pd.read_excel('KPMG2.xlsx')


# In[3]:


df.head()


# # Data cleaning

# In[4]:


df.info()


# In[4]:


df = df.drop(columns=['customer id (CustomerAddress)','Owns Car','customer id (CustomerDemographic)','Number of Records','DOB'])


# In[5]:


pd.set_option('display.max_columns',None)
df.head()


# In[6]:


df.shape


# In[9]:


df = df.rename(columns={'Job Industry Category':'category',
                        'Past 3 Years Bike Related Purchases':'past_3_yr_purchase',
                        'Product First Sold Date':'1st_sold_date',
                        'Transaction Date':'tran_date',
                        'Transaction Id':'tran_id',
                        'Customer Id':'customer_id'}
               )


# In[10]:


#yearmonth of sold date
df['month_year'] = pd.to_datetime(df['1st_sold_date']).dt.to_period('M')
df.head()


# In[11]:


# change the order of the columns
df= df[['customer_id','First Name','Last Name','Age','Gender','Job Title','own_car','Deceased Indicator',
      'Tenure','Property Valuation','category','Wealth Segment','Country','Address','Postcode','State',
      'Brand','Online Order','Order Status','tran_date','tran_id','past_3_yr_purchase',
       'Product Class','Product Id','Product Line',
      'Product Size','List Price','Standard Cost','Profit','1st_sold_date','month_year']]


# In[12]:


df.head()


# In[13]:


df.info()


# In[14]:


df.isna().sum()


# In[15]:


df = df.dropna()


# In[16]:


df.shape


# In[53]:


df.hist(bins=50, figsize=(20,15))
plt.show()


# In[18]:


cor = df.corr()
cor


# Explore correlation between the continuous feature variables.

# In[19]:


plt.figure(figsize=(12,7))
ax = sns.heatmap(cor,annot=True)
#fix the size problem
plt.ylim(10, 0)


# In[21]:


df['1st_sold_date'] = pd.to_datetime(df['1st_sold_date'])


# In[24]:


df.head()


# In[12]:


df[df['Online Order']==1.0]


# In[25]:


# drop customers who are deceased,
# first get the index, then drop
deceased = df[df['Deceased Indicator']=='Y'].index
deceased


# In[26]:


df.drop(deceased,inplace=True)


# In[27]:


# check the result
df[df['Deceased Indicator']=='Y']


# In[28]:


# final data
df.shape


# In[ ]:





# # RFM analysis

# ## Recency (R) 

# **Recency is the most important predictor of who is more likely to respond to an offer. Customers who have purchased recently are more likely to purchase again when compared to those who did not purchase recently.**

# In[30]:


#To tag the customers on the basis of recency flag, first take distinct dates of customer purchase.
sold_df = df[['customer_id','tran_date']].drop_duplicates()
sold_df


# In[57]:


# check the latest transaction date of this dataset
# sold_df['tran_date'].max()


# In[31]:


sold_df['tran_date'][0]
#type(sold_df['tran_date'][0])


# In[32]:


import datetime as dt


# In[33]:


# convert date format
sold_df['tran_date'] = pd.to_datetime(df['tran_date']).dt.strftime('%Y%m%d')
sold_df['tran_date']


# In[34]:


pd.to_numeric(sold_df['tran_date'],errors='coerce')


# In[35]:


sold_df['tran_date'] > '20171101'


# **Algorithm:  
# Tag a customer from 1 to 5 in steps of 2 months, i.e. 5 if customer bought in last 2 months; else 4 if made a purchase in last 4 months and so on. Finally for a customer, maximum of the recency flag is taken as final recency flag.**

# In[36]:


# define the recency function
def recency(row):
    if row['tran_date'] > '20171101':
        val = 5
    elif row['tran_date'] <= '20171101' and row['tran_date'] > '20170901':
        val = 4
    elif row['tran_date'] <= '20170901' and row['tran_date'] > '20170701':
        val = 3
    elif row['tran_date'] <= '20170701' and row['tran_date'] > '20170501':
        val = 2
    else:
        val = 1
    return val

# create the recency column
sold_df['Recency_Flag'] = sold_df.apply(recency, axis=1)


# In[37]:


sold_df['Recency_Flag']


# In[38]:


# For a customer, take the maximum recency flag as final recency flag
sold_df = sold_df.groupby('customer_id',as_index=False)['Recency_Flag'].max()


# In[39]:


sold_df


# In[40]:


sold_df.Recency_Flag.value_counts()


# In[77]:


# visualize the distribution of recency
sns.countplot(x='Recency_Flag',data=sold_df)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Recency_Flag', fontsize=12)
plt.xticks(rotation='vertical')
plt.title('Frequency of Recency_Flag', fontsize=15)
plt.show()


# It can be seen that more than half of the customers made a purchase in the last 2 month. Note that some customers have not visited the website in last 4-8 months.To regain that lost customer base, business should look out for the reasons why these customers stop visiting the stores.

# ## Frequency (F)

# **The second most important factor is how frequently these customers purchase. The higher the frequency, the higher is the chances of these responding to the offers.**

# **Algorithm:**  
# To tag the customers on the basis of frequency flag, first count the transaction_id for each customer, then tran_id count will be split into 5 equal parts to rank the customers on a scale of 1 to 5 where 5 being the most frequent.

# In[41]:


freq = df[['tran_id','customer_id']].drop_duplicates()
freq


# In[42]:


#Calculating the count of unique purchase for each customer
freq_count = freq.groupby(['customer_id'])['tran_id'].aggregate('count').reset_index()
freq_count 


# In[28]:


# check result for one customer
freq[freq['customer_id']==1]


# In[43]:


# rearrange the result
freq_count = freq_count.sort_values(by=['tran_id'],ascending=False) 
freq_count 


# In[44]:


# create the unique count df
unique_tranCnt = freq_count[['tran_id']].drop_duplicates()
unique_tranCnt


# In[45]:


# Dividing into 3 equal parts
unique_tranCnt['Freqency_Band'] = pd.qcut(unique_tranCnt['tran_id'], 5)
unique_tranCnt=unique_tranCnt[['Freqency_Band']].drop_duplicates()
unique_tranCnt


# Tagging customers in the range of 1 to 5 based on the count of their unique invoice where 5 corresponds to those customers who visit the store most often:

# In[32]:


# define the frequency function
def frequency(row):
    if row['tran_id'] <=4:
        val = 1
    elif row['tran_id'] >4 and row['tran_id'] <=6:
        val = 2
    elif row['tran_id'] >6 and row['tran_id'] <=9:
        val = 3
    elif row['tran_id'] >9 and row['tran_id'] <=11:
        val = 4
    else:
        val = 5
    return val

freq_count['Freq_Flag'] = freq_count.apply(frequency,axis=1)


# In[33]:


freq_count.Freq_Flag.value_counts()


# In[141]:


# visualize the distribution of frequency
sns.countplot(x='Freq_Flag',data=freq_count)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Freq_Flag', fontsize=12)
plt.xticks(rotation='vertical')
plt.title('Frequency of Freq_Flag', fontsize=15)
plt.show()


# It can be seen most of the customers purchased less than 11 times.

# ## Monetary Value (M): 

# **The third factor is the amount of money these customers have spent on purchases. Customers who have spent higher contribute more value to the business as compared to those who have spent less.**

# To tag the customers on the basis of monetary flag, first get the Total price for each customer and then 

# In[34]:


#Calculating the Sum of total monetary purchase for each customer
monetary = df.groupby(['customer_id'])['List Price'].aggregate('sum').reset_index()
monetary


# In[41]:


# reorder the result
monetary = monetary.sort_values(by='List Price',ascending=False)
monetary


# In[152]:


# check if any negative price exists
monetary[monetary['List Price']<0]


# In[35]:


# didvide total price to 5 parts
unique_price = monetary[['List Price']].drop_duplicates()
unique_price['monteray_band'] = pd.qcut(unique_price['List Price'],5)
unique_price = unique_price[['monteray_band']].drop_duplicates()
unique_price


# **Algorithm:**  
# Tagging customers in the range of 1 to 5 based on their Total price value, where 5 corresponds the customers having highest monetary value:

# In[36]:


# define the monetary function
def mo(row):
    if row['List Price'] <=3703:
        val = 1
    elif row['List Price'] >3703 and row['List Price'] <=5187:
        val = 2
    elif row['List Price'] >5187 and row['List Price'] <=6607:
        val = 3
    elif row['List Price'] >6607 and row['List Price'] <=8547:
        val = 4
    else:
        val = 5
    return val

monetary['Mon_Flag'] = monetary.apply(mo,axis=1)


# In[37]:


monetary['Mon_Flag'].value_counts()


# In[167]:


# visualize the monetary distribution
sns.countplot(x='Mon_Flag',data=monetary)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Mon_Flag', fontsize=12)
plt.xticks(rotation='vertical')
plt.title('Frequency of Mon_Flag', fontsize=15)
plt.show()


# There is an almost equal distribution of customers as far as monetary value is concerned.

# Combine all the three flags from dataframe of: sold_df, freq_count, and monetary

# In[38]:


rmf_df = pd.merge(sold_df,freq_count[['customer_id','Freq_Flag']],on=['customer_id'], how='left')


# In[39]:


rmf_df = pd.merge(rmf_df,monetary[['customer_id','Mon_Flag']],on=['customer_id'], how='left')


# In[40]:


rmf_df


# Get the combined RFM score for each customer:

# In[62]:


rmf_df[['Recency_Flag','Freq_Flag','Mon_Flag']] = rmf_df[['Recency_Flag','Freq_Flag','Mon_Flag']].astype(str)


# In[65]:


rmf_df['RMF_score'] = rmf_df[['Recency_Flag','Freq_Flag','Mon_Flag']].apply(lambda x: ''.join(x),axis=1)


# In[66]:


rmf_df 


# In[68]:


rmf_df['RMF_score'].value_counts()


# In[69]:


# export to csv file
rmf_df.to_csv('rmf.csv',index=False)


# ## Note:  
# ### Exploratory Data Analysis and further model deployment and visualization perfomed  in Tableau

# In[ ]:





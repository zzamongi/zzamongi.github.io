#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


conda config --add channels conda-forget


# In[7]:


from fbprophet import Prophet


# In[8]:


get_ipython().run_line_magic('pwd', '')


# In[9]:


df = pd.read_csv("^KS11.csv",encoding="cp949")
df


# In[10]:


data = df[['Date','Close']].reset_index(drop=True)
data = data.rename(columns={ 'Date' : 'ds', 'Close' : 'y'})
data


# In[11]:


model = Prophet(daily_seasonality = True)
model.fit(data)


# In[12]:


future = model.make_future_dataframe(periods = 365)
forecast = model.predict(future)
forecast.tail()


# In[13]:


model.plot(forecast)


# In[14]:


model.plot_components(forecast)


# In[15]:


# 3,4,9,10,11,12월 하락세, 1,2,5,6,7,8월 상승세


# In[ ]:





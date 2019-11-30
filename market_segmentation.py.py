# -*- coding: utf-8 -*-
"""
DA6823
Kilger
Exam 1
Due 10/21/18 
@author: Veronica Stephens
"""
#%%
#load libraries
import numpy as np
import pandas as pd

#%%
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

#ref:https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings#14463362

#%%
#READ IN ALL DATA
df = pd.read_feather("C:/vk/msda/08_practicum_i/simmons_market_segmentation_data_set/FA12_Data/rawsimmons.feather")
df.columns.values

#%%
#COLUMN NAMES
#get col names from data dictionary
read2='C:/vk/msda/08_practicum_i/simmons_market_segmentation_data_set/FA12_Layout_R.xlsx'
col_names = pd.read_excel(read2,header=0)
#col_names.to_csv("C:/vk/msda/08_practicum_i/exams/exam_1/all_cols.csv", sep=',')

#%%

# DEFINE FUNCTIONS


#%%
#FUNCTION: agree/disagree questions
def getagreedisagree (question,df):
#    question = "I'M 1ST OF FRNDS HAVE NEW ELCTRNC EQUIP" #debugging
    print('Question:', question)
    df_cols = col_names[col_names['level3']==question]

    #col names to select from df
    get_cols = df_cols['var_name'].tolist()
    get_cols = list(map(lambda x:x.upper(),get_cols))
    del get_cols[2] #delete any agree
    del get_cols[-1] #delete any disagree
    
    #select cols from df
    df_new = df.loc[:,get_cols]

    #for nan values replace with 0
    df_new.replace(np.nan,0,inplace=True)

    #change floats to ints
    df_new.iloc[:,0:4] = df_new.iloc[:,0:4].astype('int')

    #FREQUENCY TABLE
    names = ['agree_lot','agree_little','neither','disagree_little','disagree_lot']
    count = 0
    print()
    print('Frequency tables:')
    for col in get_cols:
        print(names[count])
        print(df_new[col].value_counts())
        print()
        count = count + 1

    #CHECK MORE THAN 1 PER ROW?
    df_new['sum']=df_new.sum(axis=1)
    print('Check if more than one answer selected:')
    print(df_new['sum'].value_counts())

    #COMBINE INTO 1 COLUMN
    #5=agree a lot, 4= agree a little, 3 = neither agree nor disagree, 
    #2 = disagree a little 1 = disagree a lot
    #order in df_new: 'agree_lot','agree_little','neither','disagree_little','disagree_lot'
    df_new['question_val'] = np.where(df_new[get_cols[0]]==1,5,
           np.where(df_new[get_cols[1]]==1,4,
                    np.where(df_new[get_cols[2]]==1,3,
                             np.where(df_new[get_cols[3]]==1,2,
                                      np.where(df_new[get_cols[4]]==1,1,0)))))
            
    #frequency new col
    freq_index = df_new['question_val'].value_counts().index.tolist()
    freq_quest = df_new['question_val'].value_counts().tolist()
    freq_dic = {}
    for i in range(0,6):
        freq_dic[freq_index[i]]=freq_quest[i]
    freq_dic.pop(0, None) #don't care about 0=didn't answer

    #validate new col= same info as 5 indivdual cols
    col_sums = df_new[get_cols].sum(axis=0)
    col_sums = col_sums[0:5].astype('int')
    print()
    print('Check if sum of columns equal to counts in combined column:')
    for i in range(0,5):
        x = 5-i
        y = i +1
        if col_sums[i] == freq_dic[x]:
            print('correct: column', y, ', value',x)
            print(col_sums[i],'=',freq_dic[x])
            print()
        else:
            print('error: column',y,', value',x)
      
    return df_new #return df

#%%
#FUCTION: get 1 column per question, combine into single df
def getquestions (list_questions,list_colnames):
#    i=1
    j=0
    df_questions = pd.DataFrame()
    for question in list_questions:
        df_working = getagreedisagree(question,df)
        df_working.rename(columns={'question_val':list_colnames[j]},inplace=True)
        df_questions[list_colnames[j]] = df_working[list_colnames[j]]
#        i = i+1
        j += 1
        print('Finished:',question);print('\n'*1)
    return df_questions
#%%
#VARIMAX FUNCTION 
#reference: https://stackoverflow.com/questions/17628589/perform-varimax-rotation-in-python-using-numpy
def varimax(Phi, gamma = 1, q = 20, tol = 1e-6):
    from numpy import eye, asarray, dot, sum, diag
    from numpy.linalg import svd
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q): #was xrange
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d/d_old < tol: break
    return dot(Phi, R)
  
#%%
    
# FACTOR ANALYSIS
  
  
#%%
#VARIABLE SELECTION STRATEGY
#first conclusion: i'm not good at selecting vars, so:
#    
#step 1
#isolate all opinion questions (agree/disagree)
#get correlations between all opinion variables
#
#step 2
#goal: to get groups of highly correlated variables for 2 factors
#randomly selected groupings of 5,4,3 variables 
#continued until threshold for KMO met (>0.7-0.8)
#
#step 3
#run pca on selected variables to see how many factors are w/in group of vars
#to get eigenvectors, variance explained
#result: begin to see paterns of which vars are clustered together
    
#step 4 
#add question groupings into longer factor analysis code below 
#to get additional: scree plot, loadings, and rotated loadings

#steps 1- 4 helped me choose the variables to use in the factor analysis
#the steps below are commented out, because it isn't needed after selecting vars

#%%
#STEP 1

##filter survey
##get df w/ all agree/disagree questions in data set
#df_opinion = col_names[col_names['level2'].str.contains(" AGREE A LOT", na=False)]
#list_questions = list(df_opinion['level3'])
#list_colnames = list(df_opinion.index.values)
#
##clean & create single col for all agree/disagree vars
#df_questions_all = getquestions(list_questions,list_colnames)
##remove rows with 0 values
##df_questions = df_questions_all[~(df_questions_all==0).any(axis=1)]
#
##get correlation matrix for all agree/disagree vars
#c = df_questions_all.corr().abs()
#s = pd.DataFrame(c.unstack())
#s['index1'] = s.index
#s=s.reset_index()
#get_cols = list(range(0,s.shape[0]))
#s['new']= np.where(s.iloc[:,2]==1,.001,s.iloc[:,2]) #don't want =1 vars, put them in the middle
#s = s.sort_values(by='new',ascending=False)
#
##add row to df for question index
#df_opinion['index2'] = df_opinion.index
#
##create vector of highly correlated and very low corr vars
##select_vars = s.iloc[np.r_[0:75,167206:167281],0] 
#
##create vector of only highly correlated vars
#select_s = s['level_0'].unique()
##select_vars = select_s[75:150]
##elements = select_s[300:350] #get uncorrelated vars for non driver vars
#select_vars = select_s[0:100]
#
##create vector with "cell" in question
##select_vars = df_opinion[df_opinion['level3'].str.contains("FIRST", na=False)]
##select_vars['index2'] = select_vars.index
##select_vars = list(select_vars.iloc[:,8])
#
##write out df_opinion to csv
#df_opinion.to_csv('C:/vk/msda/08_practicum_i/simmons_market_segmentation_data_set/df_opinion.csv', sep=',')
#
##%%
##STEP 2
#import factor_analyzer
#import random
#
##random sample from agree/disagree questions
#elements = random.sample(list(select_vars),3)
#
##creating vectors of questions to select vars from
#list_questions=[]
##list_colnames=[]
#for var in elements:
##  select_vars[0] #debugging
#  var_row = df_opinion[df_opinion['index2']==var]
#  list_questions.append(var_row.iloc[0,2])
##  list_colnames.append(var_row.iloc[0,8])
#  
#list_colnames = []
#for i in range(0,3):
#  name = 'v' + str(i)
#  list_colnames.append(name) 
#
#df_questions_test = getquestions(list_questions,list_colnames)
##remove rows with 0 values
#df_questions = df_questions_test[~(df_questions_test==0).any(axis=1)]
#
##get kmo
#df_kmo = pd.DataFrame((factor_analyzer.calculate_kmo(df_questions))[0])
#kmo_overall = (factor_analyzer.calculate_kmo(df_questions))[1]
#print();print('KMO: ');print(df_kmo)
#print();print('KMO overall score:', kmo_overall)
#
#while kmo_overall <.7 : #>.45
#  elements = random.sample(list(select_vars),3)
#  #creating vectors of questions to select vars from
#  list_questions=[]
#  #list_colnames=[]
#  for var in elements:
#  #  select_vars[0] #debugging
#    var_row = df_opinion[df_opinion['index2']==var]
#    list_questions.append(var_row.iloc[0,2])
#  #  list_colnames.append(var_row.iloc[0,8])
#    
#  list_colnames = []
#  for i in range(0,3):
#    name = 'v' + str(i)
#    list_colnames.append(name) 
#  
#  df_questions_test = getquestions(list_questions,list_colnames)
#  #remove rows with 0 values
#  df_questions = df_questions_test[~(df_questions_test==0).any(axis=1)]
#  
#  #get kmo
#  df_kmo = pd.DataFrame((factor_analyzer.calculate_kmo(df_questions))[0])
#  kmo_overall = (factor_analyzer.calculate_kmo(df_questions))[1]
#  print();print('KMO: ');print(df_kmo)
#  print();print('KMO overall score:', kmo_overall)
#
##%%
##STEP 3
##run pca 
#from sklearn.decomposition import PCA
#
##fit and transform
#n_vars = len(df_questions.columns)
#pca = PCA(n_components=n_vars)
#reduced_data_pca = pca.fit_transform(df_questions.values)
#pca_components = pca.components_
##get shape
##reduced_data_pca.shape
##pca_components.shape
#
##get eigenvalues and variance explained
#df_results = pd.DataFrame()
#df_results['eigenvalues']=pca.explained_variance_ #eigenvalues 
#df_results['prop_var']=pca.explained_variance_ratio_ #% var by each component
#df_results['cumm_var']=pca.explained_variance_ratio_.cumsum() #cummulative variance explained
#print();print('PCA results (all components):');print(df_results)

#%%
#STEP 4
#TWO QUESTION SETS (for factor analysis)
#set1 = environmentally conscious
#set2 = early adopters 

list_questions = [
    
"PEOPLE HVE RESPONS TO USE RECYCLD PRDCTS",
"COMPANIES/HELP CONSUMERS ENVRNMNT RESPNS",
"MORE LIKELY PRCH/ENVRNMNTLLY-FRNDLY COMP",
"MORE LKLY BUY/COMP W/ENVRNMNT FRNDLY ADS",

"USUALLY FRST AMNG FRNDS SHOP NEW STORE",
"I'M 1ST OF FRNDS HAVE NEW ELCTRNC EQUIP",
"I'M USUALLY FIRST TO TRY NEW HEALTH FOOD",
"I AM FRST AMNG MY FRIENDS TRY NEW STYLES",
]

#use after finalize vars
list_colnames = [
    'environRespRecycldProd',
    'enrironCompHelpConsumEnvironResp',
    'environPurchEnvironFriendlyComp',
    'environFriendlyAdPurchase',
    'firstShopNewStore',
    'firstElecEquip',
    'firstNewHealthFood',
    'firstNewStyles'] 

print('\n'*1);print('FACTOR ANALYSIS')
print('\n'*1);print('Cleaning & Compiling Variables for Factor Analysis');print('\n'*1)

#get data on 2 question sets
#clean, frequency table, combine into single column
df_questions_all = getquestions(list_questions,list_colnames)

#remove rows with 0 values
df_questions = df_questions_all[~(df_questions_all==0).any(axis=1)]

print('Question sets for factor analysis');print()
print('Set 1: Environmentally conscious')
print('Set 2: Early adopters')

#%%
#KMO & BARTLETT SPHERICITY
#ref: https://factor-analyzer.readthedocs.io/en/latest/_modules/factor_analyzer/factor_analyzer.html
import factor_analyzer

#kmo
#proportion of variance among vars may be shared
##returns: KMO score per item, KMO score overall
df_kmo = pd.DataFrame((factor_analyzer.calculate_kmo(df_questions))[0])
kmo_overall = (factor_analyzer.calculate_kmo(df_questions))[1]
print();print('KMO: ');print(df_kmo)
print();print('KMO overall score:', kmo_overall);print()
#".90s as marvelous, .80s as meritorious, .70s as middling, 
#.60s as mediocre, .50 as miserable and < .5 as unacceptable "

#bartlett spericity
#returns (chi-square value float, p-value)
print();print('Bartlett Spercicity:');print()
print('Chi-Square value:',(factor_analyzer.calculate_bartlett_sphericity(df_questions))[0])
print('P-value:',(factor_analyzer.calculate_bartlett_sphericity(df_questions))[1]);print()
#sig result = correlation matrix is different from the identity matrix, 
#data not likely malformed

#%%
#FACTOR ANALYSIS (exploratory, factors = number of vars)
#ref: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.decomposition import PCA

#fit and transform
n_vars = len(df_questions.columns)
pca = PCA(n_components=n_vars)
reduced_data_pca = pca.fit_transform(df_questions.values)
pca_components = pca.components_
#get shape
#reduced_data_pca.shape
#pca_components.shape

#get eigenvalues and variance explained
df_results = pd.DataFrame()
df_results['eigenvalues']=pca.explained_variance_ #eigenvalues 
df_results['prop_var']=pca.explained_variance_ratio_ #% var by each component
df_results['cumm_var']=pca.explained_variance_ratio_.cumsum() #cummulative variance explained
print();print('PCA results (all components):');print();print(df_results);print()

#%%
#SCREE PLOT
#ref: #https://stats.stackexchange.com/questions/12819/how-to-draw-a-scree-plot-in-python#17206
import matplotlib
import matplotlib.pyplot as plt

#Make a random array and then make it positive-definite
num_vars = n_vars
num_obs = df_questions.shape[0]
A = reduced_data_pca
A = np.asmatrix(A.T) * np.asmatrix(A)
U, S, V = np.linalg.svd(A) 
eigvals = S**2 / np.cumsum(S)[-1]

fig = plt.figure(figsize=(8,5))
sing_vals = np.arange(num_vars) + 1
plt.plot(sing_vals, eigvals, color="blue", linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')

leg = plt.legend(['Eigenvalues'], loc='best', borderpad=0.3, 
                 shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                 markerscale=0.4)
leg.get_frame().set_alpha(0.4)
leg.draggable(state=True)
plt.show()

#%%
#FACTOR ANALYSIS (repeated w/2 factors)

#fit and transform
pca = PCA(n_components=2)
reduced_data_pca = pca.fit_transform(df_questions.values)
                  
#get eigenvalues and variance explained
df_results = pd.DataFrame()
df_results['eigenvalues']=pca.explained_variance_ #eigenvalues 
df_results['prop_var']=pca.explained_variance_ratio_ #% var by each component
df_results['cumm_var']=pca.explained_variance_ratio_.cumsum() #cummulative variance explained
print();print('PCA results (2 components):');print();print(df_results);print()

#%%
#EXTRACTION COMMUNALITY
#variance in var reproduced from factors
#calc: (loadings)^2, summed across factors 

#get component loadings
#print(pca.components_.T) #eigenvectors
loadings = (pca.components_.T) * (np.sqrt(pca.explained_variance_))
#print(loadings)
#10x2 * 1x2 = 10x2 matrix

df_pca = pd.DataFrame(loadings,columns=['factor1','factor2'])
df_pca.index = list_colnames
df_pca['communality'] = (df_pca.iloc[:,0]*df_pca.iloc[:,0]) + (df_pca.iloc[:,1]*df_pca.iloc[:,1])
print();print('Extraction Communality:');print(df_pca)
print();print('Communality Estimates Total:', df_pca['communality'].sum());print()

#%%
#ROTATION
#roate original loadings
rot_comps = varimax(loadings)

#original loadings
df_orig = pd.DataFrame(loadings,index=list_colnames)
print();print('Original loadings:');print(df_orig);print()
#rotated loadings
df_ro = pd.DataFrame(rot_comps,index=list_colnames)
print();print('Rotated loadings:');print(df_ro);print()

#loading norms
#print();print("Original loadings norms:");print(np.sum(loadings**2, axis=0))
#print();print("Rotated loadings norms:");print(np.sum(rot_comps**2, axis=0))

#%%
#SAVE FACTORS

#pca.get_params(deep=True)

#type(reduced_data_pca)
reduced_data_pca.shape

#roate original data
rot_comps = pd.DataFrame(varimax(reduced_data_pca))
rot_comps.shape

#index of results df was reset during pca, need original row index
rot_comps['record'] = list(df_questions.index.values)
rot_comps['record_check'] = list(df_questions.index.values)

#join on record
df_questions['record'] = list(df_questions.index.values)
df_factors = df_questions.merge(rot_comps,on='record',how='left').set_index('record')

#drop columns
df_factors.columns.values
cols = [0,1,2,3,4,5,6,7]
df_factors.drop(df_factors.columns[cols],axis=1,inplace=True)

#rename col names
df_factors['environ'] = df_factors.iloc[:,0]
df_factors['earlyad'] = df_factors.iloc[:,1]
cols=[0,1]
df_factors.drop(df_factors.columns[cols],axis=1,inplace=True)

#index_keep = list(df_questions.index.values)

print();print('Saved Factors');print();print()


#%%

# CLUSTER ANALYSIS


#%%
##VARIABLE SELECTION: STATISTICAL/CLUSTER DRIVERS

#step 1
#isolate all opinion questions (agree/disagree)
#get correlations between all opinion variables
#
#step 2
#to get vars with low corelations correlations to 2 factors
#set correlation threshold 
#get a list of variables w/low corr to factors  to choose from

#step 3
#add drivers and run cluster analysis code below 
#to get additional: CCC, Pseudo T squared, Gap, elbow plot, cluster means

#steps 1- 3 helped me choose the variables to use in the cluster analysis
#the steps below are commented out, because it isn't needed after selecting vars

##%%
##STEP 1
##get df w/ all agree/disagree questions in data set
#df_opinion = col_names[col_names['level2'].str.contains(" AGREE A LOT", na=False)]
#list_questions = list(df_opinion['level3'])
#list_colnames = list(df_opinion.index.values)
#df_opinion['index2']=list(df_opinion.index.values)
##clean & create single col for all agree/disagree vars
#df_questions_all = getquestions(list_questions,list_colnames)
#
##create record col in df_questions_all df 
#df_questions_all['record']=list(df_questions_all.index.values)
##merge factors and all agree/disagree cleaned vars
#df_findvars = df_factors.merge(df_questions_all,on='record',how='left').set_index('record')
#
##get correlation matrix for all agree/disagree vars
#cor = df_findvars.corr().abs()
#df_cor = pd.DataFrame(cor.unstack(),columns=['cor_value'])
#df_cor['indexcopy'] = df_cor.index
#df_cor=df_cor.reset_index()
#
##%%
##STEP 2
#environ_cor = df_cor[(df_cor['level_0'] == 'environ') & (df_cor['cor_value'] <= .65)]
#earlyad_cor = df_cor[(df_cor['level_0'] == 'earlyad') & (df_cor['cor_value'] <= .65)]
#
#df_factor_cor = environ_cor.merge(earlyad_cor,on='level_1',how='inner')
#df_factor_cor.drop(0,axis=0,inplace=True)
#
#selectvars = df_factor_cor['level_1']
#selectvars = list(df_factor_cor['level_1'])
#
##creating vectors of questions to select vars from
#list_possiblevars=[]
##list_colnames=[]
#for var in selectvars:
##  selectvars[0] #debugging
#  var_row = df_opinion[df_opinion['index2']==var]
#  list_possiblevars.append(var_row.iloc[0,2])
##  list_colnames.append(var_row.iloc[0,8])

#%%
#ADD STATISTICAL DRIVERS
from sklearn import preprocessing
#people who read labels
#people who try new nutritional products

statistical_drivers = [
"WILLING TO VOLUNTEER MY TIME/GOOD CAUSE", 
"USUALLY READ INFO ON LABEL" 
]

#use after finalize vars
list_colnames = [
    'volunteerTime',
    'readLabels'] 


print('CLUSTER ANALYSIS')
print('\n'*1);print('Cleaning & Compiling Variables for Cluster Analysis');print('\n'*1)

#get data on statistical drivers
#clean, frequency table, combine into single column
df_questions_all_sd = getquestions(statistical_drivers,list_colnames)
df_questions_all_sd['index2'] = list(df_questions_all_sd.index.values)

#join 2 factors with statistical drivers 
df_factors_join = df_factors.merge(df_questions_all_sd,left_index=True,right_on='index2').set_index('record_check')
df_factors_join.drop(['index2'],axis=1,inplace=True)

#remove rows with 0 values
df_cluster = df_factors_join[~(df_factors_join==0).any(axis=1)]

print('Questions added as statistical/cluster drivers:');print()
print('WILLING TO VOLUNTEER MY TIME/GOOD CAUSE')
print('USUALLY READ INFO ON LABEL')

#%%
#KMEANS CLUSTERING
#source:#https://www.datacamp.com/community/tutorials/k-means-clustering-python#case
#ref(metrics):http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

kmeans = KMeans(n_clusters=4, max_iter=100, algorithm = 'auto',init='k-means++', random_state=0)
kmeans.fit(df_cluster)
labels = kmeans.predict(df_cluster)
centroids = kmeans.cluster_centers_
centroids = centroids.T
#centroids

wcss = []
for i in range (1,10): 
   kmeans = KMeans(n_clusters = i, init='k-means++', random_state=0) 
   kmeans.fit(df_cluster)
   wcss.append(kmeans.inertia_)
   
plt.plot(range(1,10),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()

#%%
#KMEANS CLUSTERING: CCC, PSEUDOT2
#use rpy2 package to use r packages in python (to get specific metrics)
#this took foreverrrrrrrrrrrrrr, but it works.
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()

#load r package
nbclust = importr('NbClust',lib_loc='C:/Program Files/R/R-3.5.1/library')

#run kmeans clustering
clust = nbclust.NbClust(df_cluster, min_nc=2, max_nc=10,
                        distance='euclidean',index=['ccc','pseudot2'], method="kmeans") #index='gap'

print();print('Best number of clusters (first kmeans run):')
print();print(clust.rx2("Best.nc")) #best number of clusters

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
#ref:https://rpy2.github.io/doc/latest/html/pandas.html

with localconverter(ro.default_converter + pandas2ri.converter):
  clust_stats = ro.conversion.ri2py(clust.rx2("All.index"))


#%%
#PLOT CCC, PSEUDOT2
#ref:https://jakevdp.github.io/PythonDataScienceHandbook/04.01-simple-line-plots.html

#%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

fig = plt.figure()
ax = plt.axes()
x = [2,3,4,5,6,7,8,9,10]
plt.plot(x,clust_stats[:,0])
plt.title("Cubic Cluster Criteria")
plt.xlabel("Number of Clusters")
plt.show()

plt.close()


fig = plt.figure()
ax = plt.axes()
x = [2,3,4,5,6,7,8,9,10]
plt.plot(x,clust_stats[:,1])
plt.title("Pseudo T Squared")
plt.xlabel("Number of Clusters")
plt.show()
  
plt.close()

#%%
#GAP ANALYSIS
#source: https://github.com/milesgranger/gap_statistic/blob/master/Example.ipynb
#package based on paper: https://web.stanford.edu/~hastie/Papers/gap.pdf

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from gap_statistic import OptimalK
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

optimalK = OptimalK(parallel_backend='rust')
#optimalK

test = df_cluster.as_matrix()
#type(test)

n_clusters = optimalK(test,cluster_array=np.arange(1,10))
print();print('Gap Method');print()
print('Optimal clusters: ', n_clusters)

#optimalK.gap_df.head(20)

plt.plot(optimalK.gap_df.n_clusters, optimalK.gap_df.gap_value, linewidth=3)
plt.scatter(optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].n_clusters,
            optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].gap_value, s=250, c='r')
plt.grid(True)
plt.xlabel('Cluster Count')
plt.ylabel('Gap Value')
plt.title('Gap Values by Cluster Count')
plt.show()

#%%  
#KMEANS GET CLUSTER ASSIGNMENTS

#run kmeans clustering
clust = nbclust.NbClust(df_cluster, min_nc=2, max_nc=10,
                        distance='euclidean',index=['ccc'], method="kmeans") #index='gap'

#used 4 clusters 
#get cluster assignments and add to df_cluster

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
#ref:https://rpy2.github.io/doc/latest/html/pandas.html

with localconverter(ro.default_converter + pandas2ri.converter):
  clust_assign = ro.conversion.ri2py(clust.rx2("Best.partition")) #best partition
  
df_cluster['clust_assign'] = list(clust_assign)  

#%%
#CLUSTER MEANS

cols = list(df_cluster.columns)
del cols[4]

df_centroids2 = df_cluster.groupby('clust_assign', as_index=False)[cols].mean().T
df_centroids2.columns = df_centroids2.iloc[0]
df_centroids2=df_centroids2.drop('clust_assign',axis=0)

print();print('Cluster Centers:')
print();print(df_centroids2)

#%%

# PROFILE VARIABLES

       
#%%   
#FUNCTION    
#this code is messy. these functions can be combined with functions at the start.
def getfreq (question,df,label):
#    question = 'v2507'
#    label = "SEX - HOUSEHOLD HEAD" #debugging
#    question = 'v43198'
#    label = 'RESPONDENT MOST INF IN HH PURCH DECSIONS'
    
    df_cols = col_names[col_names['level2']==label]
    label_print = df_cols[df_cols['var_name']==question].level3
    
    print('\n'*2);print('Question:', label,label_print)

    #col names to select from df
    get_cols = df_cols['var_name'].tolist()
    get_cols = list(map(lambda x:x.upper(),get_cols))
    
    #select cols from df
    df_new = df.loc[:,get_cols]
    if question == 'v43198':
      df_new = df_new.drop(df_new.columns[2],axis=1)
      del get_cols[2]      

    #FREQUENCY TABLE
    names = ['col_1','col_2']
    count = 0
    print()
    print('Frequency tables:')
    for col in get_cols:
        print(names[count])
        print(df_new[col].value_counts())
        print()
        count = count + 1

    #CHECK MORE THAN 1 PER ROW?
    df_new['sum']=df_new.sum(axis=1)
    print('Check if more than one answer selected:')
    print(df_new['sum'].value_counts())

    #COMBINE INTO 1 COLUMN
    df_new['question_val'] = np.where(df_new[get_cols[0]]==1.0,1,
           np.where(df_new[get_cols[1]]==1,0,np.nan)) #1=male,0=female
            
    #frequency new col
    freq_index = df_new['question_val'].value_counts().index.tolist()
    freq_quest = df_new['question_val'].value_counts().tolist()
    freq_dic = {}
    for i in range(0,2):
        freq_dic[freq_index[i]]=freq_quest[i]

#%%
#FUNCTION    
def getdf (question,df):
#    question='SEX - HOUSEHOLD HEAD' #debugging
#    question = 'RESPONDENT MOST INF IN HH PURCH DECSIONS'
#    question = 'CBS THIS MORNING- FREQ OF VIEWING'
  
    df_cols = col_names[col_names['level2']==question]
    
    #col names to select from df
    get_cols = df_cols['var_name'].tolist()
    get_cols = list(map(lambda x:x.upper(),get_cols))
    
    if question =='CBS THIS MORNING- FREQ OF VIEWING':
      get_cols = get_cols[0]
      df_new = df.loc[:,get_cols]
      df_new.name='mornCBS'
      df_new.replace(np.nan,0,inplace=True)
      print();print(question)
      print(df_new.value_counts())
      
    elif question == 'GOOD MORNING AMERICA (ABC)-FREQ OF VIEW':
      get_cols = get_cols[0]
      df_new = df.loc[:,get_cols]
      df_new.name='mornABC'
      df_new.replace(np.nan,0,inplace=True)
      print();print(question)
      print(df_new.value_counts())

    elif question == 'TODAY SHOW (NBC) - FREQUENCY OF VIEWING':
      get_cols = get_cols[0]
      df_new = df.loc[:,get_cols]
      df_new.name='mornNBC'
      df_new.replace(np.nan,0,inplace=True)
      print();print(question)
      print(df_new.value_counts())
    
    else:
      #select cols from df
      df_new = df.loc[:,get_cols]
      
      #COMBINE INTO 1 COLUMN
      df_new['question_val'] = np.where(df_new[get_cols[0]]==1.0,1,
             np.where(df_new[get_cols[1]]==1,0,np.nan)) #0=male,1=female
            
    return df_new #return df

#%%
#FUCTION: get 1 column per question, combine into single df
def profilequest (list_questions,list_colnames):
#    i=1
#    list_questions=profile_labels[0]
#    list_colnames=list_colnames_profile[0]
    j=0
    df_profile = pd.DataFrame()
    for question in list_questions:
#        list_questions[0]
        df_working = getdf(question,df)
        df_working.rename(columns={'question_val':list_colnames[j]},inplace=True)
        df_profile[list_colnames[j]] = df_working[list_colnames[j]]
#        i = i+1
        j += 1
        print('Finished:',question);print();print()
    return df_profile

#%%
#PROFILE VARS FREQ TABLES        
#define profile variables
profile_vars = ['v2507', #SEX - HOUSEHOLD HEAD
                'v43198' #RESPONDENT MOST INF IN HH PURCH DECSIONS
                ] 	

profile_labels = ['SEX - HOUSEHOLD HEAD',
                  'RESPONDENT MOST INF IN HH PURCH DECSIONS'
                  ]

list_colnames_profile = [
    'sex',
    'purchDecs']

print('\n'*1);print('PROFILE VARIABLES')
print('\n'*1);print('Cleaning & Compiling Profile Variables')

#check variables
getfreq(profile_vars[0],df,profile_labels[0])
getfreq(profile_vars[1],df,profile_labels[1])

df_profilevars = profilequest(profile_labels,list_colnames_profile)
df_profilevars.replace(np.nan,0,inplace=True)

#%%
#PROFILE VARS        
#define profile variables
#profile_vars = ['v17991','v17996','v18001'] 	

profile_labels = [
                  'CBS THIS MORNING- FREQ OF VIEWING',
                  'GOOD MORNING AMERICA (ABC)-FREQ OF VIEW',
                  'TODAY SHOW (NBC) - FREQUENCY OF VIEWING'
                  ]

df_cbs = pd.DataFrame(getdf(profile_labels[0],df))
df_abc = pd.DataFrame(getdf(profile_labels[1],df))
df_nbc = pd.DataFrame(getdf(profile_labels[2],df))

df_morn = df_cbs.merge(df_abc,left_index=True,right_index=True).merge(df_nbc,left_index=True,right_index=True)
#df_morn2=df_morn.rename({0:'cbs','V17996':'abc','V18001':'nbc'})

print('\n'*2)
print('Questions added as profile variables:');print()
print('SEX - HOUSEHOLD HEAD')
print('RESPONDENT MOST INF IN HH PURCH DECSIONS')
print('CBS THIS MORNING- FREQ OF VIEWING (1/week)')
print('GOOD MORNING AMERICA (ABC)-FREQ OF VIEW (1/week)')
print('TODAY SHOW (NBC) - FREQUENCY OF VIEWING (1/week)')

#%%
#COMBINE PROFILE VARIABLES

df_profile = df_morn.merge(df_profilevars,left_index=True,right_index=True)

df_final = df_cluster.merge(df_profile,how='left',left_index=True,right_index=True)
df_final.columns
cols = list(df_final.columns)
del cols[4]

df_final_means = df_final.groupby('clust_assign', as_index=False)[cols].mean().T
df_final_means.columns = df_final_means.iloc[0]
df_final_means=df_final_means.drop('clust_assign',axis=0)

print('\n'*2);print('Final Cluster Centers:')
print();print(df_final_means)

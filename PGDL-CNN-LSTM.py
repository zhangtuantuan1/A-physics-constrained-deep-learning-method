import os
import torch
from copy import deepcopy
import numpy as np
import xarray as xr
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import random
import netCDF4 as nc
from tqdm import tqdm
# from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import zipfile
import torchvision.models as models
import time
import math
from torch.autograd import grad

start = time.perf_counter()


def set_seed(seed=5000):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)


def load_data():
    # print(12)
    data1 = xr.open_dataset(r'./autodl-tmp/01-input/aa2016_2022_smap_1.nc')
    data2 = xr.open_dataset(r'./autodl-tmp/01-input/aa2016_2022_smap_2.nc')
    data3 = xr.open_dataset(r'./autodl-tmp/01-input/aa2016_2022_smap_3.nc')
    data4 = xr.open_dataset(r'./autodl-tmp/01-input/aa2016_2022_smap_4.nc')
    data5 = xr.open_dataset(r'./autodl-tmp/01-input/aa2016_2022_tp_1.nc')
    data6 = xr.open_dataset(r'./autodl-tmp/01-input/aa2016_2022_tp_2.nc')
    data7 = xr.open_dataset(r'./autodl-tmp/01-input/aa2016_2022_tp_3.nc')
    data8 = xr.open_dataset(r'./autodl-tmp/01-input/aa2016_2022_tp_4.nc')
    data9 = xr.open_dataset(r'./autodl-tmp/01-input/aa2016_2022_elev.nc')
    data10 = xr.open_dataset(r'./autodl-tmp/01-input/aa2016_2022_slope.nc')
    data11 = xr.open_dataset(r'./autodl-tmp/01-input/aa2016_2022_eva.nc')
    data12 = xr.open_dataset(r'./autodl-tmp/01-input/aa2016_2022_st.nc')
    data13 = xr.open_dataset(r'./autodl-tmp/01-input/aa2016_2022_u10.nc')
    data14 = xr.open_dataset(r'./autodl-tmp/01-input/aa2016_2022_v10.nc')
    data15 = xr.open_dataset(r'./autodl-tmp/01-input/aa2016_2022_sp.nc')
    data16 = xr.open_dataset(r'./autodl-tmp/01-input/aa2016_2022_snsr.nc')
    data17 = xr.open_dataset(r'./autodl-tmp/01-input/aa2016_2022_lai_lv.nc')
    data18 = xr.open_dataset(r'./autodl-tmp/01-input/aa2016_2022_sand.nc')
    data19 = xr.open_dataset(r'./autodl-tmp/01-input/aa2016_2022_sbd.nc')
    data20 = xr.open_dataset(r'./autodl-tmp/01-input/aa2016_2022_silt.nc')
    data21 = xr.open_dataset(r'./autodl-tmp/01-input/aa2016_2022_clay.nc')

    data22 = xr.open_dataset(r'./autodl-tmp/01-input/data_thetar.nc')
    data23 = xr.open_dataset(r'./autodl-tmp/01-input/data_thetas.nc')
    data24 = xr.open_dataset(r'./autodl-tmp/01-input/data_alpha_n.nc')
    data25 = xr.open_dataset(r'./autodl-tmp/01-input/data_ks.nc')
    data26 = xr.open_dataset(r'./autodl-tmp/01-input/data_n_n.nc')
    data27 = xr.open_dataset(r'./autodl-tmp/01-input/data_t.nc')
    data28 = xr.open_dataset(r'./autodl-tmp/01-input/data_z10.nc')



    data30 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaa2016_2022_smap_1.nc')
    data31 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaa2016_2022_smap_2.nc')
    data32 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaa2016_2022_smap_3.nc')
    data33 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaa2016_2022_smap_4.nc')
    data34 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaa2016_2022_tp_1.nc')
    data35 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaa2016_2022_tp_2.nc')
    data36 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaa2016_2022_tp_3.nc')
    data37 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaa2016_2022_tp_4.nc')
    data38 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaa2016_2022_elev.nc')
    data39 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaa2016_2022_slope.nc')
    data40 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaa2016_2022_eva.nc')
    data41 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaa2016_2022_st.nc')
    data42 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaa2016_2022_u10.nc')
    data43 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaa2016_2022_v10.nc')
    data44 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaa2016_2022_sp.nc')
    data45 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaa2016_2022_snsr.nc')
    data46 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaa2016_2022_lai_lv.nc')
    data47 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaa2016_2022_sand.nc')
    data48 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaa2016_2022_sbd.nc')
    data49 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaa2016_2022_silt.nc')
    data50 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaa2016_2022_clay.nc')

    data51 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaadata_t.nc')
    data52 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaadata_z10.nc')


    data54 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaadata_thetar.nc')
    data55 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaadata_thetas.nc')
    data56 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaadata_alpha_n.nc')
    data57 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaadata_ks.nc')
    data58 = xr.open_dataset(r'./autodl-tmp/02-input-gedian/aaadata_n_n.nc')

    data1 = data1['soil_1'][:].values
    data2 = data2['soil_2'][:].values
    data3 = data3['soil_3'][:].values
    data4 = data4['soil_4'][:].values
    data5 = data5['tp_1'][:].values
    data6 = data6['tp_2'][:].values
    data7 = data7['tp_3'][:].values
    data8 = data8['tp_4'][:].values
    data9 = data9['elev'][:].values
    data10 = data10['slope'][:].values
    data11 = data11['eva'][:].values
    data12 = data12['st'][:].values
    data13 = data13['u10'][:].values
    data14 = data14['v10'][:].values
    data15 = data15['sp'][:].values
    data16 = data16['snsr'][:].values
    data17 = data17['lai_lv'][:].values
    data18 = data18['sand'][:].values
    data19 = data19['sbd'][:].values
    data20 = data20['silt'][:].values
    data21 = data21['clay'][:].values

    data22 = data22['thetar'][:].values
    data23 = data23['thetas'][:].values
    data24 = data24['alpha_n'][:].values
    data25 = data25['ks'][:].values
    data26 = data26['n_n'][:].values
    data27 = data27['tttt'][:].values
    data28 = data28['z10'][:].values


    data30 = data30['soil_1'][:].values
    data31 = data31['soil_2'][:].values
    data32 = data32['soil_3'][:].values
    data33 = data33['soil_4'][:].values
    data34 = data34['tp_1'][:].values
    data35 = data35['tp_2'][:].values
    data36 = data36['tp_3'][:].values
    data37 = data37['tp_4'][:].values
    data38 = data38['elev'][:].values
    data39 = data39['slope'][:].values
    data40 = data40['eva'][:].values
    data41 = data41['st'][:].values
    data42 = data42['u10'][:].values
    data43 = data43['v10'][:].values
    data44 = data44['sp'][:].values
    data45 = data45['snsr'][:].values
    data46 = data46['lai_lv'][:].values
    data47 = data47['sand'][:].values
    data48 = data48['sbd'][:].values
    data49 = data49['silt'][:].values
    data50 = data50['clay'][:].values

    data51 = data51['tttt'][:].values
    data52 = data52['z10'][:].values

    data54 = data54['thetar'][:].values
    data55 = data55['thetas'][:].values
    data56 = data56['alpha_n'][:].values
    data57 = data57['ks'][:].values
    data58 = data58['n_n'][:].values

    label = xr.open_dataset(r'./autodl-tmp/02-input-gedian/shice_soil_101.nc')

    label_soil = label['soil'][:].values

    gedian_ceshi = {
        'smap_1': data30[:],
        'smap_2': data31[:],
        'smap_3': data32[:],
        'smap_4': data33[:],
        'tp_1': data34[:],
        'tp_2': data35[:],
        'tp_3': data36[:],
        'tp_4': data37[:],
        'elev': data38[:],
        'slope': data39[:],
        'eva': data40[:],
        'st': data41[:],
        'u10': data42[:],
        'v10': data43[:],
        'sp': data44[:],
        'snsr': data45[:],
        'lai_lv': data46[:],
        'sand': data47[:],
        'sbd': data48[:],
        'silt': data49[:],
        'clay': data50[:],
        'thetar': data54[:],
        'thetas': data55[:],
        'alpha_n': data56[:],
        'ks': data57[:],
        'n_n': data58[:],
        'tttt': data51[:],
        'z10': data52[:],
       }

    N = int(len(label_soil) * 0.2)
    print(N)

    dict_train1 = {
        'smap_1': data1[N:5 * N],
        'smap_2': data2[N:5 * N],
        'smap_3': data3[N:5 * N],
        'smap_4': data4[N:5 * N],
        'tp_1': data5[N:5 * N],
        'tp_2': data6[N:5 * N],
        'tp_3': data7[N:5 * N],
        'tp_4': data8[N:5 * N],
        'elev': data9[N:5 * N],
        'slope': data10[N:5 * N],
        'eva': data11[N:5 * N],
        'st': data12[N:5 * N],
        'u10': data13[N:5 * N],
        'v10': data14[N:5 * N],
        'sp': data15[N:5 * N],
        'snsr': data16[N:5 * N],
        'lai_lv': data17[N:5 * N],
        'sand': data18[N:5 * N],
        'sbd': data19[N:5 * N],
        'silt': data20[N:5 * N],
        'clay': data21[N:5 * N],
        'thetar': data22[N:5 * N],
        'thetas': data23[N:5 * N],
        'alpha_n': data24[N:5 * N],
        'ks': data25[N:5 * N],
        'n_n': data26[N:5 * N],
        'tttt': data27[N:5 * N],
        'z10': data28[N:5 * N],


        'label': label_soil[N:5 * N]}
    dict_valid1 = {
        'smap_1': data1[: N],
        'smap_2': data2[: N],
        'smap_3': data3[: N],
        'smap_4': data4[: N],
        'tp_1': data5[: N],
        'tp_2': data6[: N],
        'tp_3': data7[: N],
        'tp_4': data8[: N],
        'elev': data9[: N],
        'slope': data10[: N],
        'eva': data11[: N],
        'st': data12[: N],
        'u10': data13[: N],
        'v10': data14[: N],
        'sp': data15[: N],
        'snsr': data16[: N],
        'lai_lv': data17[: N],
        'sand': data18[: N],
        'sbd': data19[: N],
        'silt': data20[: N],
        'clay': data21[: N],
        'thetar': data22[: N],
        'thetas': data23[: N],
        'alpha_n': data24[: N],
        'ks': data25[: N],
        'n_n': data26[: N],
        'tttt': data27[: N],
        'z10': data28[: N],

        'label': label_soil[:N]}


    dict_train2 = {
        'smap_1': data1[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'smap_2': data2[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'smap_3': data3[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'smap_4': data4[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'tp_1': data5[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'tp_2': data6[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'tp_3': data7[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'tp_4': data8[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'elev': data9[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'slope': data10[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'eva': data11[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'st': data12[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'u10': data13[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'v10': data14[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'sp': data15[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'snsr': data16[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'lai_lv': data17[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'sand': data18[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'sbd': data19[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'silt': data20[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'clay': data21[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'thetar': data22[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'thetas': data23[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'alpha_n': data24[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'ks': data25[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'n_n': data26[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'tttt': data27[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],
        'z10': data28[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))],


        'label': label_soil[(list(range(0, N, 1))) + (list(range(2 * N, 5 * N, 1)))]}
    dict_valid2 = {
        'smap_1': data1[N:2* N],
        'smap_2': data2[N:2* N],
        'smap_3': data3[N:2* N],
        'smap_4': data4[N:2* N],
        'tp_1': data5[N:2* N],
        'tp_2': data6[N:2* N],
        'tp_3': data7[N:2* N],
        'tp_4': data8[N:2* N],
        'elev': data9[N:2* N],
        'slope': data10[N:2* N],
        'eva': data11[N:2* N],
        'st': data12[N:2* N],
        'u10': data13[N:2* N],
        'v10': data14[N:2* N],
        'sp': data15[N:2* N],
        'snsr': data16[N:2* N],
        'lai_lv': data17[N:2* N],
        'sand': data18[N:2* N],
        'sbd': data19[N:2* N],
        'silt': data20[N:2* N],
        'clay': data21[N:2* N],
        'thetar': data22[N:2* N],
        'thetas': data23[N:2* N],
        'alpha_n': data24[N:2* N],
        'ks': data25[N:2* N],
        'n_n': data26[N:2* N],
        'tttt': data27[N:2* N],
        'z10': data28[N:2* N],

        'label': label_soil[N:2* N]}


    dict_train3 = {
        'smap_1': data1[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'smap_2': data2[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'smap_3': data3[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'smap_4': data4[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'tp_1': data5[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'tp_2': data6[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'tp_3': data7[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'tp_4': data8[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'elev': data9[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'slope': data10[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'eva': data11[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'st': data12[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'u10': data13[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'v10': data14[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'sp': data15[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'snsr': data16[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'lai_lv': data17[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'sand': data18[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'sbd': data19[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'silt': data20[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'clay': data21[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'thetar': data22[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'thetas': data23[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'alpha_n': data24[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'ks': data25[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'n_n': data26[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'tttt': data27[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],
        'z10': data28[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))],


        'label': label_soil[(list(range(0, 2*N, 1))) + (list(range(3 * N, 5 * N, 1)))]}
    dict_valid3 = {
        'smap_1': data1[2*N:3* N],
        'smap_2': data2[2*N:3* N],
        'smap_3': data3[2*N:3* N],
        'smap_4': data4[2*N:3* N],
        'tp_1': data5[2*N:3* N],
        'tp_2': data6[2*N:3* N],
        'tp_3': data7[2*N:3* N],
        'tp_4': data8[2*N:3* N],
        'elev': data9[2*N:3* N],
        'slope': data10[2*N:3* N],
        'eva': data11[2*N:3* N],
        'st': data12[2*N:3* N],
        'u10': data13[2*N:3* N],
        'v10': data14[2*N:3* N],
        'sp': data15[2*N:3* N],
        'snsr': data16[2*N:3* N],
        'lai_lv': data17[2*N:3* N],
        'sand': data18[2*N:3* N],
        'sbd': data19[2*N:3* N],
        'silt': data20[2*N:3* N],
        'clay': data21[2*N:3* N],
        'thetar': data22[2*N:3* N],
        'thetas': data23[2*N:3* N],
        'alpha_n': data24[2*N:3* N],
        'ks': data25[2*N:3* N],
        'n_n': data26[2*N:3* N],
        'tttt': data27[2*N:3* N],
        'z10': data28[2*N:3* N],

        'label': label_soil[2*N:3* N]}


    dict_train4 = {
        'smap_1': data1[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'smap_2': data2[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'smap_3': data3[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'smap_4': data4[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'tp_1': data5[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'tp_2': data6[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'tp_3': data7[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'tp_4': data8[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'elev': data9[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'slope': data10[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'eva': data11[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'st': data12[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'u10': data13[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'v10': data14[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'sp': data15[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'snsr': data16[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'lai_lv': data17[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'sand': data18[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'sbd': data19[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'silt': data20[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'clay': data21[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'thetar': data22[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'thetas': data23[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'alpha_n': data24[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'ks': data25[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'n_n': data26[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'tttt': data27[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],
        'z10': data28[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))],


        'label': label_soil[(list(range(0, 3*N, 1))) + (list(range(4 * N, 5 * N, 1)))]}
    dict_valid4 = {
        'smap_1': data1[3*N:4* N],
        'smap_2': data2[3*N:4* N],
        'smap_3': data3[3*N:4* N],
        'smap_4': data4[3*N:4* N],
        'tp_1': data5[3*N:4* N],
        'tp_2': data6[3*N:4* N],
        'tp_3': data7[3*N:4* N],
        'tp_4': data8[3*N:4* N],
        'elev': data9[3*N:4* N],
        'slope': data10[3*N:4* N],
        'eva': data11[3*N:4* N],
        'st': data12[3*N:4* N],
        'u10': data13[3*N:4* N],
        'v10': data14[3*N:4* N],
        'sp': data15[3*N:4* N],
        'snsr': data16[3*N:4* N],
        'lai_lv': data17[3*N:4* N],
        'sand': data18[3*N:4* N],
        'sbd': data19[3*N:4* N],
        'silt': data20[3*N:4* N],
        'clay': data21[3*N:4* N],
        'thetar': data22[3*N:4* N],
        'thetas': data23[3*N:4* N],
        'alpha_n': data24[3*N:4* N],
        'ks': data25[3*N:4* N],
        'n_n': data26[3*N:4* N],
        'tttt': data27[3*N:4* N],
        'z10': data28[3*N:4* N],

        'label': label_soil[3*N:4* N]}


    dict_train5 = {
        'smap_1': data1[:4 * N],
        'smap_2': data2[:4 * N],
        'smap_3': data3[:4 * N],
        'smap_4': data4[:4 * N],
        'tp_1': data5[:4 * N],
        'tp_2': data6[:4 * N],
        'tp_3': data7[:4 * N],
        'tp_4': data8[:4 * N],
        'elev': data9[:4 * N],
        'slope': data10[:4 * N],
        'eva': data11[:4 * N],
        'st': data12[:4 * N],
        'u10': data13[:4 * N],
        'v10': data14[:4 * N],
        'sp': data15[:4 * N],
        'snsr': data16[:4 * N],
        'lai_lv': data17[:4 * N],
        'sand': data18[:4 * N],
        'sbd': data19[:4 * N],
        'silt': data20[:4 * N],
        'clay': data21[:4 * N],
        'thetar': data22[:4 * N],
        'thetas': data23[:4 * N],
        'alpha_n': data24[:4 * N],
        'ks': data25[:4 * N],
        'n_n': data26[:4 * N],
        'tttt': data27[:4 * N],
        'z10': data28[:4 * N],


        'label': label_soil[:4 * N]}
    dict_valid5 = {
        'smap_1': data1[4*N:5* N],
        'smap_2': data2[4*N:5* N],
        'smap_3': data3[4*N:5* N],
        'smap_4': data4[4*N:5* N],
        'tp_1': data5[4*N:5* N],
        'tp_2': data6[4*N:5* N],
        'tp_3': data7[4*N:5* N],
        'tp_4': data8[4*N:5* N],
        'elev': data9[4*N:5* N],
        'slope': data10[4*N:5* N],
        'eva': data11[4*N:5* N],
        'st': data12[4*N:5* N],
        'u10': data13[4*N:5* N],
        'v10': data14[4*N:5* N],
        'sp': data15[4*N:5* N],
        'snsr': data16[4*N:5* N],
        'lai_lv': data17[4*N:5* N],
        'sand': data18[4*N:5* N],
        'sbd': data19[4*N:5* N],
        'silt': data20[4*N:5* N],
        'clay': data21[4*N:5* N],
        'thetar': data22[4*N:5* N],
        'thetas': data23[4*N:5* N],
        'alpha_n': data24[4*N:5* N],
        'ks': data25[4*N:5* N],
        'n_n': data26[4*N:5* N],
        'tttt': data27[4*N:5* N],
        'z10': data28[4*N:5* N],

        'label': label_soil[4*N:5* N]}


    train_dataset1 = EarthDataSet(dict_train1)
    valid_dataset1 = EarthDataSet(dict_valid1)
    train_dataset2 = EarthDataSet(dict_train2)
    valid_dataset2 = EarthDataSet(dict_valid2)
    train_dataset3 = EarthDataSet(dict_train3)
    valid_dataset3 = EarthDataSet(dict_valid3)
    train_dataset4 = EarthDataSet(dict_train4)
    valid_dataset4 = EarthDataSet(dict_valid4)
    train_dataset5 = EarthDataSet(dict_train5)
    valid_dataset5 = EarthDataSet(dict_valid5)
    gedian_ceshi11 = EarthDataSet(gedian_ceshi)
    return train_dataset1, valid_dataset1, train_dataset2, valid_dataset2, train_dataset3, valid_dataset3, train_dataset4, valid_dataset4, train_dataset5, valid_dataset5,  gedian_ceshi11


class EarthDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['sp'])

    def __getitem__(self, idx):
        # print("aaa")
        return (self.data['smap_1'][idx], self.data['smap_2'][idx],
                self.data['smap_3'][idx], self.data['smap_4'][idx],
                self.data['tp_1'][idx], self.data['tp_1'][idx],
                self.data['tp_3'][idx], self.data['tp_4'][idx], self.data['elev'][idx],
                self.data['slope'][idx], self.data['eva'][idx], self.data['st'][idx], self.data['u10'][idx],
                self.data['v10'][idx],self.data['sp'][idx], self.data['snsr'][idx], self.data['lai_lv'][idx], self.data['sand'][idx],
                self.data['sbd'][idx], self.data['silt'][idx], self.data['clay'][idx], self.data['thetar'][idx],
                self.data['thetas'][idx],self.data['alpha_n'][idx], self.data['ks'][idx], self.data['n_n'][idx],
                self.data['tttt'][idx], self.data['z10'][idx],  self.data['label'][idx]


class CNN_LSTM(nn.Module):
    def __init__(self, n_cnn_layer: int = 1, kernals: list = [3], n_lstm_units: int = 8, n_lstm_units1: int = 4):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv2d(21, 32, kernel_size=2, stride=1, padding=0)  # 32
        self.bn = nn.BatchNorm2d(32)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=2, stride=1, padding=0)
        self.norm = nn.BatchNorm1d(1 * 1)
        self.lstm = nn.LSTM(18 * 1, n_lstm_units, 2, bidirectional=True, batch_first=True)  # dropout = 0.25
        self.linear = nn.Linear(16, 1)

    def forward(self, smap_1, smap_2, smap_3, smap_4, tp_1, tp_2, tp_3, tp_4, elev, slope, eva, st, u10, v10, sp, snsr, lai_lv, sand,
                sbd, silt, clay, thetar, thetas, alpha_n, ks, n_n, tttt, z10):
        Seqs = []
        smap1_t = smap_1[:, :, :].unsqueeze(1)
        smap2_t = smap_2[:, :, :].unsqueeze(1)
        smap3_t = smap_3[:, :, :].unsqueeze(1)
        smap4_t = smap_4[:, :, :].unsqueeze(1)
        tp_1_t = tp_1[:, :, :].unsqueeze(1)
        tp_2_t = tp_2[:, :, :].unsqueeze(1)
        tp_3_t = tp_3[:, :, :].unsqueeze(1)
        tp_4_t = tp_4[:, :, :].unsqueeze(1)
        elev_t = elev[:, :, :].unsqueeze(1)
        slope_t = slope[:, :, :].unsqueeze(1)
        eva_t = eva[:, :, :].unsqueeze(1)
        st_t = st[:, :, :].unsqueeze(1)
        u10_t = u10[:, :, :].unsqueeze(1)
        v10_t = v10[:, :, :].unsqueeze(1)
        sp_t = sp[:, :, :].unsqueeze(1)
        snsr_t = snsr[:, :, :].unsqueeze(1)
        lai_lv_t = lai_lv[:, :, :].unsqueeze(1)
        sand_t = sand[:, :, :].unsqueeze(1)
        sbd_t = sbd[:, :, :].unsqueeze(1)
        silt_t = silt[:, :, :].unsqueeze(1)
        clay_t = clay[:, :, :].unsqueeze(1)
        tttt_t = tttt[:, :].unsqueeze(1)
        z10_t = z10[:, :].unsqueeze(1)
        seq1 = torch.cat([smap1_t,smap2_t,smap3_t, smap4_t,tp_1_t,tp_2_t,tp_3_t,tp_4_t,  elev_t,slope_t,eva_t, st_t, u10_t, v10_t, sp_t, snsr_t,lai_lv_t, sand_t,sbd_t, silt_t, clay_t], dim=1)
        seq2 = self.conv1(seq1)
        seq2 = F.relu(self.bn(seq2))
        seq2 = self.conv2(seq2)
        x = seq2.squeeze(dim=3)
        x = x.permute(0, 2, 1)

        x = torch.cat([ tttt_t,z10_t, x], dim=2)
        x = self.norm(x)
        x, _ = self.lstm(x)
        # x = self.norm(x)
        x = self.linear(x)
        x = torch.sigmoid(x)
        x = 0.36 * x + 0.1

        x = x[:, :, :]
        return x

def rmse(preds, y):
    return np.sqrt(sum((preds - y) ** 2) / preds.shape[0])
def coreff(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    c1 = sum((x - x_mean) * (y - y_mean))
    c2 = sum((x - x_mean) ** 2) * sum((y - y_mean) ** 2)
    return c1 / np.sqrt(c2)

def abss(x, y):
    return sum(abs(x - y)) / x.shape[0]

fit_params = {
    'n_epochs': 50,
    'learning_rate':0.009,
    'batch_size': 1256, }

def See(theta, theta_r, theta_s):
    Se = theta
    Se = (theta - theta_r) / (theta_s - theta_r)
    Se = Se.ravel()
    return Se


def VG_dz(theta, theta_r, theta_s, n, Ks):
    Se = See(theta, theta_r, theta_s)
    m = 1 - 1 / n
    a = 1 - np.power(Se, 1 / m)
    coef = 0.5 * Ks * np.power(Se, -0.5) * np.square(1 - np.power(a, m)) + 2 * Ks * np.power(Se, -0.5 + 1 / m) * (
            1 - np.power(a, m)) * np.power(a, m - 1)
    coef = (1 / (theta_s - theta_r)) * coef
    return coef


def VG_d2z(theta, theta_r, theta_s, n, Ks, alpha):
    Se = See(theta, theta_r, theta_s)
    m = 1 - 1 / n
    a = 1 - np.power(Se, 1 / m)
    coef = np.power(Se, 0.5 - 1 / m - 1) * np.square(1 - np.power(a, m)) * np.power(np.power(Se, -1 / m),  1 / n - 1)
    coef = (Ks / ((theta_s - theta_r) * (alpha * n * m))) * coef
    return coef


def VG_dz2(theta, theta_r, theta_s, n, Ks, alpha):
    Se = See(theta, theta_r, theta_s)
    m = 1 - 1 / n
    a = 1 - np.power(Se, 1 / m)
    D = VG_d2z(theta, theta_r, theta_s, n, Ks, alpha)
    coef = (0.5 - 1 / m - 1) * np.power(Se, -1) + 2 * np.power(Se, 1 / m - 1) * np.power(1 - np.power(a, m),
                                                                                         -1) * np.power(a,
                                                                                                        m - 1) + (
                   1 / m) * (1 - 1 / n) * np.power(Se, -1 / m - 1) * np.power(np.power(Se, -1 / m) - 1, -1)
    coef = (D / (theta_s - theta_r)) * coef
    # print(coef)
    return coef


def train(train_loader, valid_loader):
    set_seed()
    model = CNN_LSTM()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = torch.optim.Adam(model.parameters(), lr=fit_params['learning_rate'])  # weight_decay=0.001
    loss_fn = nn.MSELoss()
    model.to(device)
    loss_fn.to(device)
    best_sco = 1000

    for i in range(fit_params['n_epochs']):
        print('Epoch: {}/{}'.format(i + 1, fit_params['n_epochs']))

        model.train()
        for step, ((smap_1, smap_2, smap_3, smap_4, tp_1, tp_2, tp_3, tp_4, elev, slope, eva, st, u10, v10, sp, snsr, lai_lv, sand,
                sbd, silt, clay, thetar, thetas, alpha_n, ks, n_n, tttt, z10), label) in enumerate(train_loader):

            smap_1 = smap_1.to(device).float()
            smap_2 = smap_2.to(device).float()
            smap_3 = smap_3.to(device).float()
            smap_4 = smap_4.to(device).float()
            tp_1 = tp_1.to(device).float()
            tp_2 = tp_2.to(device).float()
            tp_3 = tp_3.to(device).float()
            tp_4 = tp_4.to(device).float()
            elev = elev.to(device).float()
            slope = slope.to(device).float()
            eva = eva.to(device).float()
            st = st.to(device).float()
            u10 = u10.to(device).float()
            v10 = v10.to(device).float()
            sp = sp.to(device).float()
            snsr = snsr.to(device).float()
            lai_lv = lai_lv.to(device).float()
            sand = sand.to(device).float()
            sbd = sbd.to(device).float()
            silt = silt.to(device).float()
            clay = clay.to(device).float()

            thetar = thetar.to(device).float()
            thetas = thetas.to(device).float()
            alpha_n = alpha_n.to(device).float()
            ks = ks.to(device).float()
            n_n = n_n.to(device).float()
            tttt = tttt.to(device).float()
            z10 = z10.to(device).float()

            z10.requires_grad = True
            tttt.requires_grad = True

            optimizer.zero_grad()
            label = label.to(device).float()

            #with torch.backends.cudnn.flags(enabled=False):

            preds = model(smap_1, smap_2, smap_3, smap_4, tp_1, tp_2, tp_3, tp_4, elev, slope, eva, st, u10, v10, sp, snsr, lai_lv, sand,
                sbd, silt, clay, thetar, thetas, alpha_n, ks, n_n, tttt, z10)

            preds11 = preds.squeeze(dim=2)
            len11 = preds.shape[0]

            ######
            u_x = grad(preds11.sum(), z10, create_graph=True)[0]
            u_t = grad(preds11.sum(), tttt, create_graph=True)[0]
            u_xx = grad(u_x.sum(), z10, create_graph=True)[0]

            jj1 = []
            jj2 = []
            jj3 = []
            for jj in range(len11):
                theta_r = thetar[jj, 0].cpu().detach().numpy()
                # print(type(theta_r))
                theta_s = thetas[jj, 0].cpu().detach().numpy()
                ks_1 = ks[jj, 0].cpu().detach().numpy()
                n_n_1 = n_n[jj, 0].cpu().detach().numpy()
                alpha_n_1 = alpha_n[jj, 0].cpu().detach().numpy()
                u = preds[jj, 0].cpu().detach().numpy()
                u11 = preds[jj, 0]

                f1 = VG_dz(u, theta_r, theta_s, n_n_1, ks_1)
                f2 = VG_d2z(u, theta_r, theta_s, n_n_1, ks_1, alpha_n_1)
                f3 = VG_dz2(u, theta_r, theta_s, n_n_1, ks_1, alpha_n_1)
                jj1.append(f1)
                jj2.append(f2)
                jj3.append(f3)

            jj1_array = np.array(jj1)
            jj2_array = np.array(jj2)
            jj3_array = np.array(jj3)
            jj1_tensor = torch.tensor(jj1_array.squeeze())
            jj1_tensor = jj1_tensor.to(u_x.device)
            jj1_tensor = jj1_tensor.unsqueeze(-1)

            jj2_tensor = torch.tensor(jj2_array.squeeze())
            jj2_tensor = jj2_tensor.to(u_x.device)
            jj2_tensor = jj2_tensor.unsqueeze(-1)
            jj3_tensor = torch.tensor(jj3_array.squeeze())
            jj3_tensor = jj3_tensor.to(u_x.device)
            jj3_tensor = jj3_tensor.unsqueeze(-1)
            aaa1 = u_t
            aaa2 = jj1_tensor * u_x + jj2_tensor * u_xx + jj3_tensor * (torch.square(u_x))

            loss11 = loss_fn(aaa1, aaa2)
            loss22 = loss_fn(preds11, label)
            loss =  loss22+loss11
            loss.backward()
            optimizer.step()

            if step % 1000 == 0:
                print('loss {}'.format(loss))

    with torch.no_grad():
        model.eval()
        y_true, y_pred, y_yubao = [], [], []
        for step, (
                (smap_1, smap_2, smap_3, smap_4, tp_1, tp_2, tp_3, tp_4, elev, slope, eva, st, u10, v10, sp, snsr, lai_lv, sand,
                sbd, silt, clay, thetar, thetas, alpha_n, ks, n_n, tttt, z10),label) in enumerate(valid_loader):

            smap_1 = smap_1.to(device).float()
            smap_2 = smap_2.to(device).float()
            smap_3 = smap_3.to(device).float()
            smap_4 = smap_4.to(device).float()
            tp_1 = tp_1.to(device).float()
            tp_2 = tp_2.to(device).float()
            tp_3 = tp_3.to(device).float()
            tp_4 = tp_4.to(device).float()
            elev = elev.to(device).float()
            slope = slope.to(device).float()
            eva = eva.to(device).float()
            st = st.to(device).float()
            u10 = u10.to(device).float()
            v10 = v10.to(device).float()
            sp = sp.to(device).float()
            snsr = snsr.to(device).float()
            lai_lv = lai_lv.to(device).float()
            sand = sand.to(device).float()
            sbd = sbd.to(device).float()
            silt = silt.to(device).float()
            clay = clay.to(device).float()

            thetar = thetar.to(device).float()
            thetas = thetas.to(device).float()
            alpha_n = alpha_n.to(device).float()
            ks = ks.to(device).float()
            n_n = n_n.to(device).float()
            tttt = tttt.to(device).float()
            z10 = z10.to(device).float()

            preds = model(smap_1, smap_2, smap_3, smap_4, tp_1, tp_2, tp_3, tp_4, elev, slope, eva, st, u10, v10, sp, snsr, lai_lv, sand,
                sbd, silt, clay, thetar, thetas, alpha_n, ks, n_n, tttt, z10)
            preds = preds.squeeze(dim=2)
            preds_pre = smap_1[:, 1, 1].unsqueeze(1)
            y_pred.append(preds)
            y_true.append(label)
            y_yubao.append(preds_pre)
        y_true1 = torch.cat(y_true, axis=0)
        y_pred1 = torch.cat(y_pred, axis=0)
        y_yubao1 = torch.cat(y_yubao, axis=0)


    return y_true1, y_pred1, y_yubao1

train_dataset1, valid_dataset1, train_dataset2, valid_dataset2, train_dataset3, valid_dataset3, train_dataset4, valid_dataset4, train_dataset5, valid_dataset5,gedian_ceshi11 = load_data()
train_loader1 = DataLoader(train_dataset1, batch_size=fit_params['batch_size'], shuffle=True, pin_memory=True,
                           num_workers=3)
valid_loader1 = DataLoader(valid_dataset1, batch_size=fit_params['batch_size'], shuffle=False, pin_memory=True,
                           num_workers=3)
train_loader2 = DataLoader(train_dataset2, batch_size=fit_params['batch_size'], shuffle=True, pin_memory=True,
                           num_workers=3)
valid_loader2 = DataLoader(valid_dataset2, batch_size=fit_params['batch_size'], shuffle=False, pin_memory=True,
                           num_workers=3)
train_loader3 = DataLoader(train_dataset3, batch_size=fit_params['batch_size'], shuffle=True, pin_memory=True,
                           num_workers=3)
valid_loader3 = DataLoader(valid_dataset3, batch_size=fit_params['batch_size'], shuffle=False, pin_memory=True,
                           num_workers=3)
train_loader4 = DataLoader(train_dataset4, batch_size=fit_params['batch_size'], shuffle=True, pin_memory=True,
                           num_workers=3)
valid_loader4 = DataLoader(valid_dataset4, batch_size=fit_params['batch_size'], shuffle=False, pin_memory=True,
                           num_workers=3)
train_loader5 = DataLoader(train_dataset5, batch_size=fit_params['batch_size'], shuffle=True, pin_memory=True,
                           num_workers=3)
valid_loader5 = DataLoader(valid_dataset5, batch_size=fit_params['batch_size'], shuffle=False, pin_memory=True,
                           num_workers=3)
gedian_loader11 = DataLoader(gedian_ceshi11, batch_size=fit_params['batch_size'], shuffle=False, pin_memory=True,
                           num_workers=3)

if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()

    y_true1, y_pred1, y_yubao1 = train(train_loader1, valid_loader1)
    print("1")
    y_true2, y_pred2, y_yubao2 = train(train_loader2, valid_loader2)
    print("2")
    y_true3, y_pred3, y_yubao3 = train(train_loader3, valid_loader3)
    print("3")
    y_true4, y_pred4, y_yubao4 = train(train_loader4, valid_loader4)
    print("4")
    y_true5, y_pred5, y_yubao5 = train(train_loader5, valid_loader5)
    print("5")
    y_true111, y_pred111, y_yubao111 = train(train_loader5, gedian_loader11)
    print(y_pred111.shape)
    y_pred111 = y_pred111.cpu().detach().numpy().astype(np.float32)
    y_pred222 = y_pred111.reshape(2557, 16, 27)
    np.save('y_pred_z10.npy', y_pred222)
    y_true = torch.cat((y_true1, y_true2, y_true3, y_true4, y_true5),axis=0)
    y_pred = torch.cat((y_pred1, y_pred2, y_pred3, y_pred4, y_pred5 ), axis=0)
    y_yubao = torch.cat((y_yubao1, y_yubao2, y_yubao3, y_yubao4, y_yubao5), axis=0)

    y_true = y_true.cpu().detach().numpy().astype(np.float32)
    y_pred = y_pred.cpu().detach().numpy().astype(np.float32)
    y_yubao = y_yubao.cpu().detach().numpy().astype(np.float32)

    for i in range(y_pred.shape[0]):
        if y_pred[i] < 0:
            y_pred[i] = 0
    print(max(y_pred))
    print(min(y_pred))
    print(max(y_true))
    print(max(y_yubao))

    np.save('y_true_z10_23.npy', y_true)
    np.save('y_pred_z10_23.npy', y_pred)
    np.save('y_yubao_z10_23.npy', y_yubao)

    end = time.perf_counter()
    run_time = (end - start) / 60
    print("运行时间：", run_time, "分")
#




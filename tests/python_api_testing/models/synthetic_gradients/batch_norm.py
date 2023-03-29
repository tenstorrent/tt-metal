import math

import torch
import numpy as np
from pymetal import ttmetal as ttm
from utility_functions import tt2torch, tilize_to_list

def batchnorm1d_inference(weight, bias, running_mean, running_var, epsilon: float, L: int, device):

    gamma = ttm.tensor.Tensor(weight, [1, 1, 32, L], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)
    beta = ttm.tensor.Tensor(bias, [1, 1, 32, L], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)
    epsilon = ttm.tensor.Tensor([epsilon] + [0 for _ in range(32 * 32 - 1)], [1, 1, 32, 32], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)
    running_var = ttm.tensor.Tensor(running_var, [1, 1, 32, L], ttm.tensor.DataType.BFLOAT16,ttm.tensor.Layout.TILE, device)
    running_mean = ttm.tensor.Tensor(running_mean, [1, 1, 32, L], ttm.tensor.DataType.BFLOAT16,ttm.tensor.Layout.TILE, device)


    BCHW = ttm.tensor.BcastOpDim.HW
    BCADD = ttm.tensor.BcastOpMath.ADD

    def batchnorm1d_inference_(X):
        var_plus_eps = ttm.tensor.bcast(running_var, epsilon, BCADD, BCHW)
        sqrt_var = ttm.tensor.sqrt(var_plus_eps)
        sqrt_inv = ttm.tensor.recip(sqrt_var)
        x_minus_mean = ttm.tensor.sub(X, running_mean)
        x_div_sqrt = ttm.tensor.mul(x_minus_mean, sqrt_inv)
        x_gamma = ttm.tensor.mul(x_div_sqrt, gamma)
        Y = ttm.tensor.add(x_gamma, beta)
        return Y

    return batchnorm1d_inference_

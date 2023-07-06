import torch
import tt_lib

def tt_linear(x, weight, bias=None):
    #weight = tt_lib.tensor.transpose(weight)
    x = tt_lib.tensor.matmul(x, weight)
    if bias is not None:
        x = tt_lib.tensor.bcast(
            x, bias, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H
        )
    return x

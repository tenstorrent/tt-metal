import torch
import numpy as np
from loguru import logger

import tt_lib


def get_shape(shape):
    """Insert 1's in the begining of shape list until the len(shape) = 4"""
    if len(shape) <= 4:
        new_shape = [1 for i in range(4 - len(shape))]
        new_shape.extend(shape)
    else:
        new_shape = shape
    return new_shape


def tt_linear(weight: tt_lib.tensor, bias: tt_lib.tensor, device):
    """Perform a linear operation on the input tensor using transposed weight and bias."""

    def linear_(activation):
        weight_T = tt_lib.tensor.transpose(weight)
        output = tt_lib.tensor.matmul(activation, weight_T)

        if bias is not None:
            output_plus_bias = tt_lib.tensor.bcast(
                output, bias, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H
            )
            return output_plus_bias

        return output

    return linear_


def is_torch_tensor(x):
    if type(x) is torch.Tensor:
        return True
    else:
        return False

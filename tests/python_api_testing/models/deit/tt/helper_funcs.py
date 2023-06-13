from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")

import torch
from torch import nn

from typing import Set, List, Tuple


import tt_lib
from tt_lib import tensor
from utility_functions_new import  torch_to_tt_tensor_rm


def Linear(in_features: int, out_features: int, weight, bias):
    """
    Returns a function that performs a Linear operation with optional bias.

    ``weight`` must be the weight as a tilized list of values.
    """

    weight = weight
    bias = bias

    def linear_(activation):
        weight_T = tensor.transpose(weight)
        output = tensor.matmul(activation, weight_T)

        if bias is not None:
            output_plus_bias = tensor.bcast(output, bias, tensor.BcastOpMath.ADD, tensor.BcastOpDim.H)
            return output_plus_bias

        return output

    return linear_


def make_linear(in_features: int, out_features: int, weights, bias, device):
    weights = torch_to_tt_tensor_rm(weights, device, shape=[1, 1, out_features, in_features], put_on_device=False)
    bias = torch_to_tt_tensor_rm(bias, device, shape=[1, 1, 1, out_features], put_on_device=False) if bias is not None else None
    return Linear(in_features, out_features, weights, bias)

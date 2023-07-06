import torch
from torch.nn import functional as F

import tt_lib
from tt_lib.fallback_ops import fallback_ops

import math

from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)

def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def tt_nanogpt_gelu(x, device):
    z = x

    tt_k1 = fallback_ops.full(x.shape(), 0.5)

    tt_k2 = fallback_ops.full(x.shape(), 0.044715)

    tt_k3 = fallback_ops.full(x.shape(), 2.0)

    tt_k4 = fallback_ops.full(x.shape(), math.pi)

    tt_k4_recip = tt_lib.tensor.recip(tt_k4)

    new_factor = tt_lib.tensor.mul(tt_k3, tt_k4_recip)

    factor1 = tt_lib.tensor.mul(tt_k1, z)

    pow2 = tt_lib.tensor.mul(z, z)

    pow3 = tt_lib.tensor.mul(pow2, z)
    factor3 = tt_lib.tensor.mul(tt_k2, pow3)

    factor3 = tt_lib.tensor.add(factor3, z)

    sumtanh = tt_lib.tensor.mul(new_factor, factor3)
    tanh = tt_lib.tensor.tanh(sumtanh)

    tt_k5 = fallback_ops.full(x.shape(), 1.0)

    total = tt_lib.tensor.add(tt_k5, tanh)
    output = tt_lib.tensor.mul(factor1, total)

    return output

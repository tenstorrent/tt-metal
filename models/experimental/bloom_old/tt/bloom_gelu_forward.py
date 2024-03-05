# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
import torch
import math
from torch.nn import functional as F

import tt_lib as ttm
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
import numpy as np
import models.experimental.bloom_old.bloom_utils as bloom_utils


def bloom_gelu_forward(x: torch.Tensor) -> torch.Tensor:
    """
    Custom bias GELU function. Adapted from Megatron-DeepSpeed code. Here we use a simple implementation (inference) to
    make the model jitable.

    Args:
        x (`torch.tensor`, *required*):
            input hidden states
    """
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def tt_bloom_gelu_forward(x, device):
    z = x

    k1 = torch.full(tuple(x.get_legacy_shape()), 0.5)
    tt_k1 = bloom_utils.torch2tt_tensor(k1, device)

    k2 = torch.full(tuple(x.get_legacy_shape()), 0.044715)
    tt_k2 = bloom_utils.torch2tt_tensor(k2, device)

    k3 = torch.full(tuple(x.get_legacy_shape()), 0.79788456)
    tt_k3 = bloom_utils.torch2tt_tensor(k3, device)

    # 0.5*x
    factor1 = ttm.tensor.mul(tt_k1, z)  # exp(z)

    # x*x
    pow2 = ttm.tensor.mul(z, z)

    # (x + 0.044715 * torch.pow(x, 3)))
    # torch.pow(x, 3))
    pow3 = ttm.tensor.mul(pow2, z)
    factor3 = ttm.tensor.mul(tt_k2, pow3)

    # (x + 0.044715 * torch.pow(x, 3)))
    factor3 = ttm.tensor.add(factor3, z)

    sumtanh = ttm.tensor.mul(tt_k3, factor3)
    tanh = ttm.tensor.tanh(sumtanh)

    k4 = torch.full(tuple(x.get_legacy_shape()), 1.0)
    tt_k4 = bloom_utils.torch2tt_tensor(k4, device)

    total = ttm.tensor.add(tt_k4, tanh)
    output = ttm.tensor.mul(factor1, total)

    return output

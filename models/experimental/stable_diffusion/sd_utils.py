# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.utility_functions import torch_to_tt_tensor_rm

from models.helper_funcs import Linear as SDLinear


def make_linear(
    in_features: int, out_features: int, weights: ttnn.Tensor, bias: ttnn.Tensor, device, out_mem_config=None
) -> SDLinear:
    if out_mem_config is None:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    weights = torch_to_tt_tensor_rm(weights, device, shape=[1, 1, out_features, in_features], put_on_device=False)
    bias = (
        torch_to_tt_tensor_rm(bias, device, shape=[1, 1, 1, out_features], put_on_device=False)
        if bias is not None
        else None
    )
    return SDLinear(in_features, out_features, weights, bias)

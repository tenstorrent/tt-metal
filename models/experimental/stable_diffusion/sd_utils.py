# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch

import tt_lib as ttl
from models.utility_functions import torch_to_tt_tensor_rm

from models.helper_funcs import Linear as SDLinear


def make_linear(
    in_features: int,
    out_features: int,
    weights: ttl.tensor.Tensor,
    bias: ttl.tensor.Tensor,
    device,
    out_mem_config=None,
) -> SDLinear:
    if out_mem_config is None:
        out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)

    weights = torch_to_tt_tensor_rm(weights, device, shape=[1, 1, out_features, in_features], put_on_device=False)
    bias = (
        torch_to_tt_tensor_rm(bias, device, shape=[1, 1, 1, out_features], put_on_device=False)
        if bias is not None
        else None
    )
    return SDLinear(in_features, out_features, weights, bias)

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import ttnn


def Linear(
    in_features: int,
    out_features: int,
    weight: ttnn.Tensor,
    bias: Optional[ttnn.Tensor] = None,
    output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
):
    """
    Returns a function that performs a Linear operation with optional bias.

    ``weight`` must be tt_tensor.
    """
    assert weight.padded_shape == [
        1,
        1,
        out_features,
        in_features,
    ], "weight does not have the expected shape"

    if bias is not None:
        assert bias.padded_shape[-1] == out_features, "bias does not have the expected shape"

    weight = weight
    bias = bias
    # Use permute instead of transpose to avoid segfault on some configurations
    # permute requires tensor to be on device, so check storage type
    if weight.device() is None:
        # Weight is on host, use torch transpose on the underlying data
        weight_torch = ttnn.to_torch(weight)
        weight_T_torch = weight_torch.transpose(-2, -1).contiguous()
        weight_T = ttnn.from_torch(weight_T_torch, dtype=weight.dtype, layout=weight.layout)
    else:
        weight_T = ttnn.permute(weight, (0, 1, 3, 2))

    def linear_(activation):
        nonlocal bias
        assert activation.padded_shape[-1] == in_features, "activation tensor do not have the expected shape"
        if bias is not None and bias.get_layout() != ttnn.TILE_LAYOUT:
            bias = ttnn.to_layout(bias, ttnn.TILE_LAYOUT)
        return ttnn.linear(activation, weight_T, bias=bias, memory_config=output_mem_config)

    return linear_

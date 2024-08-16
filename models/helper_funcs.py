# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Optional
import ttnn
from loguru import logger


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
    assert weight.get_legacy_shape() == [1, 1, out_features, in_features], "weight does not have the expected shape"

    if bias is not None:
        assert bias.get_legacy_shape()[-1] == out_features, "bias does not have the expected shape"

    weight = weight
    bias = bias
    weight_T = ttnn.transpose(weight, -2, -1)

    def linear_(activation):
        nonlocal bias
        assert activation.get_legacy_shape()[-1] == in_features, "activation tensor do not have the expected shape"
        if bias is not None and bias.get_layout() != ttnn.TILE_LAYOUT:
            bias = ttnn.to_layout(bias, ttnn.TILE_LAYOUT)
        return ttnn.linear(activation, weight_T, bias=bias, memory_config=output_mem_config)

    return linear_

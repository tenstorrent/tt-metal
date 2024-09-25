# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Optional
from ttnn import matmul
import ttnn


def Linear(
    in_features: int,
    out_features: int,
    weight: ttnn.Tensor,
    bias: Optional[ttnn.Tensor],
    device,
):
    """
    Returns a function that performs a Linear operation with optional bias.

    ``weight`` must be the weight as a tilized list of values.
    """
    assert weight.shape.with_tile_padding() == [1, 1, out_features, in_features]

    if bias is None:
        bias = None
    else:
        assert bias.shape.with_tile_padding() == [1, 1, 32, out_features]

    if bias is not None and bias.get_layout() != ttnn.TILE_LAYOUT:
        bias = ttnn.to_layout(bias, ttnn.TILE_LAYOUT)

    def linear_(activation):
        weight_T = ttnn.transpose(weight, -2, -1)
        return ttnn.linear(activation, weight_T, bias=bias)

    return linear_

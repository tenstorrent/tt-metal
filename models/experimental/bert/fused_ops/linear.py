# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Optional
from tt_lib import tensor
from ttnn import matmul
import ttnn


def Linear(
    in_features: int,
    out_features: int,
    weight: tensor.Tensor,
    bias: Optional[tensor.Tensor],
    device,
):
    """
    Returns a function that performs a Linear operation with optional bias.

    ``weight`` must be the weight as a tilized list of values.
    """
    assert weight.get_legacy_shape() == [1, 1, out_features, in_features]

    if bias is None:
        bias = None
    else:
        assert bias.get_legacy_shape() == [1, 1, 32, out_features]

    def linear_(activation):
        nonlocal bias
        weight_T = ttnn.transpose(weight, -2, -1)
        if bias is not None and bias.get_layout() != ttnn.TILE_LAYOUT:
            bias = ttnn.to_layout(bias, ttnn.TILE_LAYOUT)

        return ttnn.linear(activation, weight_T, bias=bias)

    return linear_

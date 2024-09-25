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
    # weight = ttnn.Tensor(
    #     weight,
    #     [1, 1, out_features, in_features],
    #     ttnn.bfloat16,
    #     ttnn.TILE_LAYOUT,
    #     device
    # )

    if bias is None:
        bias = None
    else:
        assert bias.shape.with_tile_padding() == [1, 1, 32, out_features]
        # bias = ttnn.Tensor(
        #     bias,
        #     [1, 1, 32, out_features],
        #     ttnn.bfloat16,
        #     ttnn.TILE_LAYOUT,
        #     device
        # )

    def linear_(activation):
        weight_T = ttnn.transpose(weight, -2, -1)
        output = ttnn.matmul(activation, weight_T)

        if bias is not None:
            output_plus_bias = ttnn.add(output, bias)
            return output_plus_bias

        return output

    return linear_

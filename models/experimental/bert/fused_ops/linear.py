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
        weight_T = ttnn.transpose(weight, -2, -1)
        output = ttnn.matmul(activation, weight_T)

        if bias is not None:
            output_plus_bias = tensor.bcast(output, bias, tensor.BcastOpMath.ADD, tensor.BcastOpDim.H)
            return output_plus_bias

        return output

    return linear_

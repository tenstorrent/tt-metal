# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Optional
from tt_lib import tensor
import tt_lib as ttl
from loguru import logger
import tt_lib


def Linear(
    in_features: int,
    out_features: int,
    weight: tensor.Tensor,
    bias: Optional[tensor.Tensor] = None,
    output_mem_config=tt_lib.tensor.MemoryConfig(
        tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
    ),
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
    weight_T = tensor.transpose(weight, -2, -1)

    def linear_(activation):
        assert activation.get_legacy_shape()[-1] == in_features, "activation tensor do not have the expected shape"
        output = tensor.matmul(activation, weight_T, output_mem_config)

        if bias is not None:
            output_plus_bias = tensor.bcast(
                output, bias, tensor.BcastOpMath.ADD, tensor.BcastOpDim.H, output_mem_config
            )
            return output_plus_bias

        return output

    return linear_

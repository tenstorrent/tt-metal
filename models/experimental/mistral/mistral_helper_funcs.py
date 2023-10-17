# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import tt_lib
from typing import Optional
from models.utility_functions import tt_to_torch_tensor, torch_to_tt_tensor_rm


def Linear(
    in_features: int,
    out_features: int,
    weight: tt_lib.tensor.Tensor,
    bias: Optional[tt_lib.tensor.Tensor] = None,
    output_mem_config=tt_lib.tensor.MemoryConfig(
        tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
    ),
    device=None,
):
    """
    Returns a function that performs a Linear operation with optional bias.

    ``weight`` must be tt_tensor.
    """
    assert weight.shape() == [1, 1, out_features, in_features], "weight does not have the expected shape"

    if bias is not None:
        assert bias.shape()[-1] == out_features, "bias does not have the expected shape"

    weight = weight
    bias = bias
    weight = tt_to_torch_tensor(weight)
    weight_T = weight.transpose(2, 3)
    weight_T = torch_to_tt_tensor_rm(weight_T, device, put_on_device=False)

    def linear_(activation):
        assert activation.shape()[-1] == in_features, "activation tensor do not have the expected shape"
        output = tt_lib.tensor.matmul(activation, weight_T, output_mem_config)

        if bias is not None:
            output_plus_bias = tt_lib.tensor.bcast(
                output, bias, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H, output_mem_config
            )
            return output_plus_bias

        return output

    return linear_

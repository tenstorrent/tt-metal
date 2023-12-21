# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import tt_lib
import ttnn
from typing import Optional


def Linear(
    weight: tt_lib.tensor.Tensor,
    bias: Optional[tt_lib.tensor.Tensor] = None,
    device=None,
    output_mem_config=tt_lib.tensor.MemoryConfig(
        tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
    ),
):
    weight_T = ttnn.permute(weight, (1, 0))

    def linear_(activation):
        output = ttnn.matmul(activation, weight_T)
        if bias is not None:
            output_plus_bias = ttnn.add(
                output,
                bias,
                alpha=1,
            )
            return output_plus_bias
        return output

    return linear_

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import ttnn.deprecated
import ttnn
from typing import Optional


def Linear(
    weight: ttnn.experimental.tensor.Tensor,
    bias: Optional[ttnn.experimental.tensor.Tensor] = None,
    device=None,
    output_mem_config=ttnn.experimental.tensor.MemoryConfig(
        ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.DRAM
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

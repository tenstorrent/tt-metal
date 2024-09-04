# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import Optional


def Linear(
    weight: ttnn.Tensor, bias: Optional[ttnn.Tensor] = None, device=None, output_mem_config=ttnn.DRAM_MEMORY_CONFIG
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

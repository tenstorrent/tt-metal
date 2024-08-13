# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn.deprecated
import ttnn


def linear(x, weight, bias=None):
    out_mem_config_l1 = ttnn.experimental.tensor.MemoryConfig(
        ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.L1
    )

    weight = ttnn.transpose(weight, -2, -1)
    x = ttnn.matmul(x, weight)
    if bias is not None:
        x = ttnn.experimental.tensor.bcast(
            x, bias, ttnn.experimental.tensor.BcastOpMath.ADD, ttnn.experimental.tensor.BcastOpDim.H
        )
    return x

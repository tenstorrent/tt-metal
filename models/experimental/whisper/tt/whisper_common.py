# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
import ttnn


def linear(x, weight, bias=None):
    out_mem_config_l1 = tt_lib.tensor.MemoryConfig(
        tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1
    )

    weight = ttnn.transpose(weight, -2, -1)
    x = ttnn.matmul(x, weight)
    if bias is not None:
        x = ttnn.add(x, bias)
    return x

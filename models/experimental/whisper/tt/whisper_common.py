# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
import ttnn


def linear(x, weight, bias=None):
    out_mem_config_l1 = tt_lib.tensor.MemoryConfig(
        tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1
    )

    weight = tt_lib.tensor.transpose(weight, -2, -1)
    x = ttnn.matmul(x, weight)
    if bias is not None:
        x = tt_lib.tensor.bcast(x, bias, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H)
    return x

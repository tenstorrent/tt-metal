# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


def linear(x, weight, bias=None):
    out_mem_config_l1 = ttnn.L1_MEMORY_CONFIG

    weight = ttnn.transpose(weight, -2, -1)
    x = ttnn.matmul(x, weight)
    if bias is not None:
        x = ttnn.add(x, bias)
    return x

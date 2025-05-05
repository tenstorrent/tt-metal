# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
import ttnn


# *****************************************************************************
# PyTorch model
# *****************************************************************************


class TestMatmulModel1(torch.nn.Module):
    def __init__(self):
        super(TestMatmulModel1, self).__init__()

    def forward(self, x1, x2):
        return torch.matmul(x1, x2)


# *****************************************************************************
# TTNN model
# *****************************************************************************


class TestMatmulModel1TTNN:
    def __init__(self):
        pass

    def __call__(self, in0, in1):
        return ttnn.matmul(in0, in1)


# *****************************************************************************
# Optimized TTNN model
# *****************************************************************************


class TestMatmulModel1TTNNOptimized:
    def __init__(self, device):
        self.device = device

    def __call__(self, in0, in1):
        # Extract input data
        # Input 0
        in0 = None
        dtype0 = None
        mem_config0 = None

        # Input 1
        in1 = None
        dtype1 = None
        mem_config1 = None

        # Layer 0: input
        in0_t = torch2tt_tensor(in0, self.device, tt_memory_config=mem_config0, tt_dtype=dtype0)
        in1_t = torch2tt_tensor(in1, self.device, tt_memory_config=mem_config1, tt_dtype=dtype1)

        # Layer 1: matmul, output
        mm = ttnn.matmul(in0_t, in1_t)
        out = tt2torch_tensor(mm)

        return out

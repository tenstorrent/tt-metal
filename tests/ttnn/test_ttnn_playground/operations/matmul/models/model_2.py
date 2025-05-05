# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
import ttnn


# *****************************************************************************
# PyTorch model
# *****************************************************************************


class TestMatmulModel2(torch.nn.Module):
    def __init__(self):
        super(TestMatmulModel2, self).__init__()

    def forward(self, x1, x2):
        # Layer 1: matmul
        mm1 = torch.matmul(x1, x2)
        mm2 = torch.matmul(x1, x2)

        # Layer 2: matmul
        mm3 = torch.matmul(mm1, mm2)

        return mm3


# *****************************************************************************
# TTNN model
# *****************************************************************************


class TestMatmulModel2TTNN:
    def __init__(self):
        pass

    def __call__(self, in0, in1):
        # Layer 1: matmul
        mm1 = ttnn.matmul(in0, in1)
        mm2 = ttnn.matmul(in0, in1)

        # Layer 2: matmul, output
        mm3 = ttnn.matmul(mm1, mm2)

        return mm3


# *****************************************************************************
# Optimized TTNN model
# *****************************************************************************


class TestMatmulModel2TTNNOptimized:
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

        # Layer 1: matmul
        mm1 = ttnn.matmul(in0_t, in1_t)
        mm2 = ttnn.matmul(in0_t, in1_t)

        # Layer 2: matmul, output
        mm3 = ttnn.matmul(mm1, mm2)
        out = tt2torch_tensor(mm3)

        return out

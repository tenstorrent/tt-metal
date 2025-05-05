# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
import ttnn


# *****************************************************************************
# PyTorch model
# *****************************************************************************


class TestMatmulModel3(torch.nn.Module):
    def __init__(self):
        super(TestMatmulModel3, self).__init__()

    def forward(self, x1, x2, x3, x4):
        # Layer 1: matmul
        mm1 = torch.matmul(x1, x2)
        mm2 = torch.matmul(x2, x3)
        mm3 = torch.matmul(x3, x4)

        # Layer 2: matmul
        mm4 = torch.matmul(mm1, mm2)
        mm5 = torch.matmul(mm2, mm3)

        # Layer 3: matmul
        mm6 = torch.matmul(mm4, mm5)

        return mm6


# *****************************************************************************
# TTNN model
# *****************************************************************************


class TestMatmulModel3TTNN:
    def __init__(self):
        pass

    def __call__(self, in0, in1, in2, in3):
        # Layer 1: matmul
        mm1 = ttnn.matmul(in0, in1)
        mm2 = ttnn.matmul(in1, in2)
        mm3 = ttnn.matmul(in2, in3)

        # Layer 2: matmul
        mm4 = ttnn.matmul(mm1, mm2)
        mm5 = ttnn.matmul(mm2, mm3)

        # Layer 3: matmul, output
        mm6 = ttnn.matmul(mm4, mm5)

        return mm6


# *****************************************************************************
# Optimized TTNN model
# *****************************************************************************


class TestMatmulModel3TTNNOptimized:
    def __init__(self, device):
        self.device = device

    def __call__(self, in0, in1, in2, in3):
        # Extract input data
        # Input 0
        in0 = None
        dtype0 = None
        mem_config0 = None

        # Input 1
        in1 = None
        dtype1 = None
        mem_config1 = None

        # Input 2
        in2 = None
        dtype2 = None
        mem_config2 = None

        # Input 3
        in3 = None
        dtype3 = None
        mem_config3 = None

        # Layer 0: input
        in0_t = torch2tt_tensor(in0, self.device, tt_memory_config=mem_config0, tt_dtype=dtype0)
        in1_t = torch2tt_tensor(in1, self.device, tt_memory_config=mem_config1, tt_dtype=dtype1)
        in2_t = torch2tt_tensor(in2, self.device, tt_memory_config=mem_config2, tt_dtype=dtype2)
        in3_t = torch2tt_tensor(in3, self.device, tt_memory_config=mem_config3, tt_dtype=dtype3)

        # Layer 1: matmul
        mm1 = ttnn.matmul(in0_t, in1_t)
        mm2 = ttnn.matmul(in1_t, in2_t)
        mm3 = ttnn.matmul(in2_t, in3_t)

        # Layer 2: matmul
        mm4 = ttnn.matmul(mm1, mm2)
        mm5 = ttnn.matmul(mm2, mm3)

        # Layer 3: matmul, output
        mm6 = ttnn.matmul(mm4, mm5)
        out = tt2torch_tensor(mm6)

        return out

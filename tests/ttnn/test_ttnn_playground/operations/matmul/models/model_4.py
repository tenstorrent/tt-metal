# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
import ttnn


# *****************************************************************************
# PyTorch model
# *****************************************************************************


class TestMatmulModel4(torch.nn.Module):
    def __init__(self):
        super(TestMatmulModel4, self).__init__()

    def forward(self, x1, x2, x3, x4):
        # Layer 1: matmul
        mm1 = torch.matmul(x1, x2)
        mm2 = torch.matmul(x2, x3)
        mm3 = torch.matmul(x3, x4)

        # Layer 2: matmul
        mm4 = torch.matmul(mm2, mm1)
        mm5 = torch.matmul(mm1, mm2)
        mm6 = torch.matmul(mm1, mm3)
        mm7 = torch.matmul(mm2, mm3)

        # Layer 3: matmul
        mm8 = torch.matmul(mm4, mm5)
        mm9 = torch.matmul(mm5, mm6)
        mm10 = torch.matmul(mm6, mm7)

        # Layer 4: matmul
        mm11 = torch.matmul(mm8, mm9)
        mm12 = torch.matmul(mm9, mm10)

        # Layer 5: matmul
        mm13 = torch.matmul(mm11, mm12)

        return mm13


# *****************************************************************************
# TTNN model
# *****************************************************************************


class TestMatmulModel4TTNN:
    def __init__(self):
        pass

    def __call__(self, in0, in1, in2, in3):
        # Layer 1: matmul
        mm1 = ttnn.matmul(in0, in1)
        mm2 = ttnn.matmul(in1, in2)
        mm3 = ttnn.matmul(in2, in3)

        # Layer 2: matmul
        mm4 = ttnn.matmul(mm2, mm1)
        mm5 = ttnn.matmul(mm1, mm2)
        mm6 = ttnn.matmul(mm1, mm3)
        mm7 = ttnn.matmul(mm2, mm3)

        # Layer 3: matmul
        mm8 = ttnn.matmul(mm4, mm5)
        mm9 = ttnn.matmul(mm5, mm6)
        mm10 = ttnn.matmul(mm6, mm7)

        # Layer 4: matmul
        mm11 = ttnn.matmul(mm8, mm9)
        mm12 = ttnn.matmul(mm9, mm10)

        # Layer 5: matmul, output
        mm13 = ttnn.matmul(mm11, mm12)

        return mm13


# *****************************************************************************
# Optimized TTNN model
# *****************************************************************************


class TestMatmulModel4TTNNOptimized:
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
        in0_t = torch2tt_tensor(in0, dtype0, mem_config0, self.device)
        in1_t = torch2tt_tensor(in1, dtype1, mem_config1, self.device)
        in2_t = torch2tt_tensor(in2, dtype2, mem_config2, self.device)
        in3_t = torch2tt_tensor(in3, dtype3, mem_config3, self.device)

        # Layer 1: matmul
        mm1 = ttnn.matmul(in0_t, in1_t)
        mm2 = ttnn.matmul(in1_t, in2_t)
        mm3 = ttnn.matmul(in2_t, in3_t)

        # Layer 2: matmul
        mm4 = ttnn.matmul(mm2, mm1)
        mm5 = ttnn.matmul(mm1, mm2)
        mm6 = ttnn.matmul(mm1, mm3)
        mm7 = ttnn.matmul(mm2, mm3)

        # Layer 3: matmul
        mm8 = ttnn.matmul(mm4, mm5)
        mm9 = ttnn.matmul(mm5, mm6)
        mm10 = ttnn.matmul(mm6, mm7)

        # Layer 4: matmul
        mm11 = ttnn.matmul(mm8, mm9)
        mm12 = ttnn.matmul(mm9, mm10)

        # Layer 5: matmul, output
        mm13 = ttnn.matmul(mm11, mm12)
        out = tt2torch_tensor(mm13)

        return out

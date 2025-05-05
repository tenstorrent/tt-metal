# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
import ttnn


# *****************************************************************************
# PyTorch model
# *****************************************************************************


class TestMatmulModel7(torch.nn.Module):
    def __init__(self):
        super(TestMatmulModel7, self).__init__()

    def forward(self, x1, x2, x3, x4):
        # Layer 1: matmul
        mm1 = torch.matmul(x1, x2)
        mm2 = torch.matmul(x2, x3)
        mm3 = torch.matmul(x3, x4)

        # Layer 2: matmul
        mm4 = torch.matmul(mm1, mm2)
        mm5 = torch.matmul(mm2, mm3)
        mm6 = torch.matmul(mm2, mm3)

        # Layer 3: matmul
        mm7 = torch.matmul(mm4, mm2)
        mm8 = torch.matmul(mm1, mm5)
        mm9 = torch.matmul(mm5, mm6)

        # Layer 4: matmul
        mm10 = torch.matmul(mm1, mm7)
        mm11 = torch.matmul(mm7, mm8)
        mm12 = torch.matmul(mm5, mm9)

        # Layer 5: matmul
        mm13 = torch.matmul(mm10, mm11)
        mm14 = torch.matmul(mm8, mm12)

        # Layer 6: matmul
        mm15 = torch.matmul(mm13, mm11)
        mm16 = torch.matmul(mm11, mm14)

        # Layer 7: matmul
        mm17 = torch.matmul(mm15, mm16)

        return mm17


# *****************************************************************************
# TTNN model
# *****************************************************************************


class TestMatmulModel7TTNN:
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
        mm6 = ttnn.matmul(mm2, mm3)

        # Layer 3: matmul
        mm7 = ttnn.matmul(mm4, mm2)
        mm8 = ttnn.matmul(mm1, mm5)
        mm9 = ttnn.matmul(mm5, mm6)

        # Layer 4: matmul
        mm10 = ttnn.matmul(mm1, mm7)
        mm11 = ttnn.matmul(mm7, mm8)
        mm12 = ttnn.matmul(mm5, mm9)

        # Layer 5: matmul
        mm13 = ttnn.matmul(mm10, mm11)
        mm14 = ttnn.matmul(mm8, mm12)

        # Layer 6: matmul
        mm15 = ttnn.matmul(mm13, mm11)
        mm16 = ttnn.matmul(mm11, mm14)

        # Layer 7: matmul
        mm17 = ttnn.matmul(mm15, mm16)

        return mm17


# *****************************************************************************
# Optimized TTNN model
# *****************************************************************************

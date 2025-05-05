# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
import ttnn


# *****************************************************************************
# PyTorch model
# *****************************************************************************


class TestMatmulModel9(torch.nn.Module):
    def __init__(self):
        super(TestMatmulModel9, self).__init__()

    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
        # Layer 1: matmul
        mm1 = torch.matmul(x1, x2)
        mm2 = torch.matmul(x2, x6)
        mm3 = torch.matmul(x6, x7)

        # Layer 2: matmul
        mm4 = torch.matmul(x3, mm1)
        mm5 = torch.matmul(mm1, mm2)
        mm6 = torch.matmul(mm2, mm3)
        mm7 = torch.matmul(mm3, x8)

        # Layer 3: matmul
        mm8 = torch.matmul(mm5, mm6)
        mm9 = torch.matmul(x4, mm4)
        mm10 = torch.matmul(mm9, mm8)
        mm11 = torch.matmul(mm7, x9)
        mm12 = torch.matmul(mm8, mm11)

        # Layer 4: matmul
        mm13 = torch.matmul(x5, mm9)
        mm14 = torch.matmul(mm11, x10)

        # Layer 5: matmul
        mm15 = torch.matmul(mm10, mm12)

        # Layer 6: matmul
        mm16 = torch.matmul(mm13, mm15)
        mm17 = torch.matmul(mm15, mm14)

        # Layer 7: matmul
        mm18 = torch.matmul(mm16, mm17)

        return mm18


# *****************************************************************************
# TTNN model
# *****************************************************************************


class TestMatmulModel9TTNN:
    def __init__(self):
        pass

    def __call__(self, in0, in1, in2, in3, in4, in5, in6, in7, in8, in9):
        # Layer 1: matmul
        mm1 = ttnn.matmul(in0, in1)
        mm2 = ttnn.matmul(in1, in5)
        mm3 = ttnn.matmul(in5, in6)

        # Layer 2: matmul
        mm4 = ttnn.matmul(in2, mm1)
        mm5 = ttnn.matmul(mm1, mm2)
        mm6 = ttnn.matmul(mm2, mm3)
        mm7 = ttnn.matmul(mm3, in7)

        # Layer 3: matmul
        mm8 = ttnn.matmul(mm5, mm6)
        mm9 = ttnn.matmul(in3, mm4)
        mm10 = ttnn.matmul(mm9, mm8)
        mm11 = ttnn.matmul(mm7, in8)
        mm12 = ttnn.matmul(mm8, mm11)

        # Layer 4: matmul
        mm13 = ttnn.matmul(in4, mm9)
        mm14 = ttnn.matmul(mm11, in9)

        # Layer 5: matmul
        mm15 = ttnn.matmul(mm10, mm12)

        # Layer 6: matmul
        mm16 = ttnn.matmul(mm13, mm15)
        mm17 = ttnn.matmul(mm15, mm14)

        # Layer 7: matmul, output
        mm18 = ttnn.matmul(mm16, mm17)

        return mm18


# *****************************************************************************
# Optimized TTNN model
# *****************************************************************************

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
import ttnn


# *****************************************************************************
# PyTorch model
# *****************************************************************************


class TestMatmulModel8(torch.nn.Module):
    def __init__(self):
        super(TestMatmulModel8, self).__init__()

    def forward(self, x1, x2, x3, x4, x5, x6):
        # Layer 1: matmul
        # Flow 1: LEFT
        mm1 = torch.matmul(x1, x2)
        mm2 = torch.matmul(x2, x3)
        mm3 = torch.matmul(x2, x3)

        # Flow 2: RIGHT
        mm4 = torch.matmul(x4, x5)
        mm5 = torch.matmul(x4, x5)
        mm6 = torch.matmul(x5, x6)

        # Layer 2: matmul
        mm7 = torch.matmul(x1, mm1)
        mm8 = torch.matmul(mm1, x2)
        mm9 = torch.matmul(mm1, mm2)
        mm10 = torch.matmul(mm3, mm4)
        mm11 = torch.matmul(x5, mm6)
        mm12 = torch.matmul(mm6, x6)

        # Layer 3: matmul
        mm13 = torch.matmul(mm7, mm8)
        mm14 = torch.matmul(mm8, mm9)
        mm15 = torch.matmul(mm9, mm10)
        mm16 = torch.matmul(mm10, mm5)
        mm17 = torch.matmul(mm5, mm11)
        mm18 = torch.matmul(mm17, mm12)

        # Layer 4: matmul
        mm19 = torch.matmul(mm13, mm14)
        mm20 = torch.matmul(mm14, mm15)
        mm21 = torch.matmul(mm15, mm16)
        mm22 = torch.matmul(mm17, mm18)

        # Layer 5: matmul
        mm23 = torch.matmul(mm13, mm19)
        mm24 = torch.matmul(mm19, mm20)
        mm25 = torch.matmul(mm20, mm21)
        mm26 = torch.matmul(mm21, mm22)

        # Layer 6: matmul
        mm27 = torch.matmul(mm23, mm24)
        mm28 = torch.matmul(mm24, mm25)
        mm29 = torch.matmul(mm25, mm26)
        mm30 = torch.matmul(mm26, mm18)

        # Layer 7: matmul
        mm31 = torch.matmul(mm27, mm24)
        mm32 = torch.matmul(mm28, mm29)
        mm33 = torch.matmul(mm29, mm30)

        # Layer 8: matmul
        mm34 = torch.matmul(mm31, mm28)
        mm35 = torch.matmul(mm32, mm33)

        # Layer 9: matmul
        mm36 = torch.matmul(mm34, mm35)

        return mm36


# *****************************************************************************
# TTNN model
# *****************************************************************************


class TestMatmulModel8TTNN:
    def __init__(self):
        pass

    def __call__(self, in0, in1, in2, in3, in4, in5):
        # Layer 1: matmul
        # Flow 1: LEFT
        mm1 = ttnn.matmul(in0, in1)
        mm2 = ttnn.matmul(in1, in2)
        mm3 = ttnn.matmul(in1, in2)

        # Flow 2: RIGHT
        mm4 = ttnn.matmul(in3, in4)
        mm5 = ttnn.matmul(in3, in4)
        mm6 = ttnn.matmul(in4, in5)

        # Layer 2: matmul
        mm7 = ttnn.matmul(in0, mm1)
        mm8 = ttnn.matmul(mm1, in1)
        mm9 = ttnn.matmul(mm1, mm2)
        mm10 = ttnn.matmul(mm3, mm4)
        mm11 = ttnn.matmul(in4, mm6)
        mm12 = ttnn.matmul(mm6, in5)

        # Layer 3: matmul
        mm13 = ttnn.matmul(mm7, mm8)
        mm14 = ttnn.matmul(mm8, mm9)
        mm15 = ttnn.matmul(mm9, mm10)
        mm16 = ttnn.matmul(mm10, mm5)
        mm17 = ttnn.matmul(mm5, mm11)
        mm18 = ttnn.matmul(mm17, mm12)

        # Layer 4: matmul
        mm19 = ttnn.matmul(mm13, mm14)
        mm20 = ttnn.matmul(mm14, mm15)
        mm21 = ttnn.matmul(mm15, mm16)
        mm22 = ttnn.matmul(mm17, mm18)

        # Layer 5: matmul
        mm23 = ttnn.matmul(mm13, mm19)
        mm24 = ttnn.matmul(mm19, mm20)
        mm25 = ttnn.matmul(mm20, mm21)
        mm26 = ttnn.matmul(mm21, mm22)

        # Layer 6: matmul
        mm27 = ttnn.matmul(mm23, mm24)
        mm28 = ttnn.matmul(mm24, mm25)
        mm29 = ttnn.matmul(mm25, mm26)
        mm30 = ttnn.matmul(mm26, mm18)

        # Layer 7: matmul
        mm31 = ttnn.matmul(mm27, mm24)
        mm32 = ttnn.matmul(mm28, mm29)
        mm33 = ttnn.matmul(mm29, mm30)

        # Layer 8: matmul
        mm34 = ttnn.matmul(mm31, mm28)
        mm35 = ttnn.matmul(mm32, mm33)

        # Layer 9: matmul, output
        mm36 = ttnn.matmul(mm34, mm35)

        return mm36


# *****************************************************************************
# Optimized TTNN model
# *****************************************************************************

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
import ttnn


# *****************************************************************************
# PyTorch model
# *****************************************************************************


class TestMatmulModel6(torch.nn.Module):
    def __init__(self):
        super(TestMatmulModel6, self).__init__()

    def forward(self, x1, x2, x3, x4, x5, x6):
        # Layer 1: matmul
        mm1 = torch.matmul(x1, x2)
        mm2 = torch.matmul(x2, x3)
        mm3 = torch.matmul(x3, x4)
        mm4 = torch.matmul(x4, x5)
        mm5 = torch.matmul(x5, x6)

        # Layer 2: matmul
        mm6 = torch.matmul(mm1, mm2)
        mm7 = torch.matmul(mm2, mm3)
        mm8 = torch.matmul(mm4, mm5)

        # Layer 3: matmul
        mm9 = torch.matmul(mm1, mm6)
        mm10 = torch.matmul(mm6, mm7)
        mm11 = torch.matmul(mm3, mm8)
        mm12 = torch.matmul(mm8, mm5)

        # Layer 4: matmul
        mm13 = torch.matmul(mm9, mm10)
        mm14 = torch.matmul(mm10, mm11)
        mm15 = torch.matmul(mm11, mm12)

        # Layer 5: matmul
        mm16 = torch.matmul(mm9, mm13)
        mm17 = torch.matmul(mm13, mm14)
        mm18 = torch.matmul(mm14, mm15)
        mm19 = torch.matmul(mm15, mm12)

        # Layer 6: matmul
        mm20 = torch.matmul(mm16, mm13)
        mm21 = torch.matmul(mm17, mm14)
        mm22 = torch.matmul(mm15, mm19)

        # Layer 7: matmul
        mm23 = torch.matmul(mm20, mm17)
        mm24 = torch.matmul(mm21, mm18)
        mm25 = torch.matmul(mm22, mm12)

        # Layer 8: matmul
        mm26 = torch.matmul(mm20, mm23)
        mm27 = torch.matmul(mm24, mm25)

        # Layer 9: matmul
        mm28 = torch.matmul(mm26, mm27)

        return mm28


# *****************************************************************************
# TTNN model
# *****************************************************************************


class TestMatmulModel6TTNN:
    def __init__(self):
        pass

    def __call__(self, in0, in1, in2, in3, in4, in5):
        # Layer 1: matmul
        mm1 = ttnn.matmul(in0, in1)
        mm2 = ttnn.matmul(in1, in2)
        mm3 = ttnn.matmul(in2, in3)
        mm4 = ttnn.matmul(in3, in4)
        mm5 = ttnn.matmul(in4, in5)

        # Layer 2: matmul
        mm6 = ttnn.matmul(mm1, mm2)
        mm7 = ttnn.matmul(mm2, mm3)
        mm8 = ttnn.matmul(mm4, mm5)

        # Layer 3: matmul
        mm9 = ttnn.matmul(mm1, mm6)
        mm10 = ttnn.matmul(mm6, mm7)
        mm11 = ttnn.matmul(mm3, mm8)
        mm12 = ttnn.matmul(mm8, mm5)

        # Layer 4: matmul
        mm13 = ttnn.matmul(mm9, mm10)
        mm14 = ttnn.matmul(mm10, mm11)
        mm15 = ttnn.matmul(mm11, mm12)

        # Layer 5: matmul
        mm16 = ttnn.matmul(mm9, mm13)
        mm17 = ttnn.matmul(mm13, mm14)
        mm18 = ttnn.matmul(mm14, mm15)
        mm19 = ttnn.matmul(mm15, mm12)

        # Layer 6: matmul
        mm20 = ttnn.matmul(mm16, mm13)
        mm21 = ttnn.matmul(mm17, mm14)
        mm22 = ttnn.matmul(mm15, mm19)

        # Layer 7: matmul
        mm23 = ttnn.matmul(mm20, mm17)
        mm24 = ttnn.matmul(mm21, mm18)
        mm25 = ttnn.matmul(mm22, mm19)

        # Layer 8: matmul
        mm26 = ttnn.matmul(mm20, mm23)
        mm27 = ttnn.matmul(mm24, mm25)

        # Layer 9: matmul
        mm28 = ttnn.matmul(mm26, mm27)

        return mm28


# *****************************************************************************
# Optimized TTNN model
# *****************************************************************************

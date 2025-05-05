# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
import ttnn


# *****************************************************************************
# PyTorch model
# *****************************************************************************


class TestMatmulModel10(torch.nn.Module):
    def __init__(self, layer=9):
        super(TestMatmulModel10, self).__init__()
        if layer not in range(1, 10):
            raise ValueError("Layer must be between 1 and 9")
        self.layer = layer

    def forward(self, x1, x2, x3, x4, x5, x6):
        # Layer 1: matmul
        mm1 = torch.matmul(x1, x2)
        mm2 = torch.matmul(x1, x2)
        mm3 = torch.matmul(x3, x4)
        mm4 = torch.matmul(x3, x4)
        mm5 = torch.matmul(x5, x6)
        mm6 = torch.matmul(x5, x6)

        if self.layer == 1:
            return mm1, mm2, mm3, mm4, mm5, mm6

        # Layer 2: matmul
        mm7 = torch.matmul(x1, mm1)
        mm8 = torch.matmul(mm1, mm2)
        mm9 = torch.matmul(x3, mm3)
        mm10 = torch.matmul(mm3, mm4)
        mm11 = torch.matmul(x5, mm5)
        mm12 = torch.matmul(mm5, mm6)

        if self.layer == 2:
            return mm7, mm8, mm9, mm10, mm11, mm12

        # Layer 3: matmul
        mm13 = torch.matmul(mm7, mm8)
        mm14 = torch.matmul(mm9, mm10)
        mm15 = torch.matmul(mm11, mm12)

        if self.layer == 3:
            return mm13, mm14, mm15

        # Layer 4: matmul
        mm16 = torch.matmul(mm13, mm14)
        mm17 = torch.matmul(mm13, mm14)
        mm18 = torch.matmul(mm14, mm15)
        mm19 = torch.matmul(mm14, mm15)

        if self.layer == 4:
            return mm16, mm17, mm18, mm19

        # Layer 5: matmul
        mm20 = torch.matmul(mm13, mm16)
        mm21 = torch.matmul(mm16, mm17)
        mm22 = torch.matmul(mm14, mm18)
        mm23 = torch.matmul(mm18, mm19)

        if self.layer == 5:
            return mm20, mm21, mm22, mm23

        # Layer 6: matmul
        mm24 = torch.matmul(mm20, mm21)
        mm25 = torch.matmul(mm22, mm23)

        if self.layer == 6:
            return mm24, mm25

        # Layer 7: matmul
        mm26 = torch.matmul(mm24, mm25)
        mm27 = torch.matmul(mm24, mm25)

        if self.layer == 7:
            return mm26, mm27

        # Layer 8: matmul
        mm28 = torch.matmul(mm24, mm26)
        mm29 = torch.matmul(mm26, mm27)

        if self.layer == 8:
            return mm28, mm29

        # Layer 9: matmul
        mm30 = torch.matmul(mm28, mm29)

        return mm30


# *****************************************************************************
# TTNN model
# *****************************************************************************


class TestMatmulModel10TTNN:
    def __init__(self, layer=9):
        if layer not in range(1, 10):
            raise ValueError("Layer must be between 1 and 9")
        self.layer = layer

    def __call__(self, in0, in1, in2, in3, in4, in5):
        # Layer 1: matmul
        mm1 = ttnn.matmul(in0, in1)
        mm2 = ttnn.matmul(in0, in1)
        mm3 = ttnn.matmul(in2, in3)
        mm4 = ttnn.matmul(in2, in3)
        mm5 = ttnn.matmul(in4, in5)
        mm6 = ttnn.matmul(in4, in5)

        if self.layer == 1:
            return mm1, mm2, mm3, mm4, mm5, mm6

        # Layer 2: matmul
        mm7 = ttnn.matmul(in0, mm1)
        mm8 = ttnn.matmul(mm1, mm2)
        mm9 = ttnn.matmul(in2, mm3)
        mm10 = ttnn.matmul(mm3, mm4)
        mm11 = ttnn.matmul(in4, mm5)
        mm12 = ttnn.matmul(mm5, mm6)

        if self.layer == 2:
            return mm7, mm8, mm9, mm10, mm11, mm12

        # Layer 3: matmul
        mm13 = ttnn.matmul(mm7, mm8)
        mm14 = ttnn.matmul(mm9, mm10)
        mm15 = ttnn.matmul(mm11, mm12)

        if self.layer == 3:
            return mm13, mm14, mm15

        # Layer 4: matmul
        mm16 = ttnn.matmul(mm13, mm14)
        mm17 = ttnn.matmul(mm13, mm14)
        mm18 = ttnn.matmul(mm14, mm15)
        mm19 = ttnn.matmul(mm14, mm15)

        if self.layer == 4:
            return mm16, mm17, mm18, mm19

        # Layer 5: matmul
        mm20 = ttnn.matmul(mm13, mm16)
        mm21 = ttnn.matmul(mm16, mm17)
        mm22 = ttnn.matmul(mm14, mm18)
        mm23 = ttnn.matmul(mm18, mm19)

        if self.layer == 5:
            return mm20, mm21, mm22, mm23

        # Layer 6: matmul
        mm24 = ttnn.matmul(mm20, mm21)
        mm25 = ttnn.matmul(mm22, mm23)

        if self.layer == 6:
            return mm24, mm25

        # Layer 7: matmul
        mm26 = ttnn.matmul(mm24, mm25)
        mm27 = ttnn.matmul(mm24, mm25)

        if self.layer == 7:
            return mm26, mm27

        # Layer 8: matmul
        mm28 = ttnn.matmul(mm24, mm26)
        mm29 = ttnn.matmul(mm26, mm27)

        if self.layer == 8:
            return mm28, mm29

        # Layer 9: matmul, output
        mm30 = ttnn.matmul(mm28, mm29)

        return mm30


# *****************************************************************************
# Optimized TTNN model
# *****************************************************************************

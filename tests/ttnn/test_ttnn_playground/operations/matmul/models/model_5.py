# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
import ttnn


# *****************************************************************************
# PyTorch model
# *****************************************************************************


class TestMatmulModel5(torch.nn.Module):
    def __init__(self):
        super(TestMatmulModel5, self).__init__()

    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13):
        # Layer 1: matmul
        mm1 = torch.matmul(x1, x2)
        mm2 = torch.matmul(x2, x3)
        mm3 = torch.matmul(x3, x4)

        # Layer 2: matmul
        mm4 = torch.matmul(x5, mm1)
        mm5 = torch.matmul(mm2, x11)
        mm6 = torch.matmul(mm3, x8)

        # Layer 3: matmul
        mm7 = torch.matmul(x6, mm4)
        mm8 = torch.matmul(mm5, x12)
        mm9 = torch.matmul(mm6, x9)

        # Layer 4: matmul
        mm10 = torch.matmul(x7, mm7)
        mm11 = torch.matmul(mm8, x13)
        mm12 = torch.matmul(mm9, x10)

        # Layer 5: matmul
        mm13 = torch.matmul(mm10, mm11)
        mm14 = torch.matmul(mm11, mm12)

        # Layer 6: matmul
        mm15 = torch.matmul(mm13, mm14)

        return mm15


# *****************************************************************************
# TTNN model
# *****************************************************************************


class TestMatmulModel5TTNN:
    def __init__(self):
        pass

    def __call__(
        self,
        in0,
        in1,
        in2,
        in3,
        in4,
        in5,
        in6,
        in7,
        in8,
        in9,
        in10,
        in11,
        in12,
    ):
        # Layer 1: matmul
        mm1 = ttnn.matmul(in0, in1)
        mm2 = ttnn.matmul(in1, in2)
        mm3 = ttnn.matmul(in2, in3)

        # Layer 2: matmul
        mm4 = ttnn.matmul(in4, mm1)
        mm5 = ttnn.matmul(mm2, in10)
        mm6 = ttnn.matmul(mm3, in7)

        # Layer 3: matmul
        mm7 = ttnn.matmul(in5, mm4)
        mm8 = ttnn.matmul(mm5, in11)
        mm9 = ttnn.matmul(mm6, in8)

        # Layer 4: matmul
        mm10 = ttnn.matmul(in6, mm7)
        mm11 = ttnn.matmul(mm8, in12)
        mm12 = ttnn.matmul(mm9, in9)

        # Layer 5: matmul
        mm13 = ttnn.matmul(mm10, mm11)
        mm14 = ttnn.matmul(mm11, mm12)

        # Layer 6: matmul
        mm15 = ttnn.matmul(mm13, mm14)

        return mm15


# *****************************************************************************
# Optimized TTNN model
# *****************************************************************************


class TestMatmulModel5TTNNOptimized:
    def __init__(self, device):
        self.device = device

    def __call__(self, in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12):
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

        # Input 4
        in4 = None
        dtype4 = None
        mem_config4 = None

        # Input 5
        in5 = None
        dtype5 = None
        mem_config5 = None

        # Input 6
        in6 = None
        dtype6 = None
        mem_config6 = None

        # Input 7
        in7 = None
        dtype7 = None
        mem_config7 = None

        # Input 8
        in8 = None
        dtype8 = None
        mem_config8 = None

        # Input 9
        in9 = None
        dtype9 = None
        mem_config9 = None

        # Input 10
        in10 = None
        dtype10 = None
        mem_config10 = None

        # Input 11
        in11 = None
        dtype11 = None
        mem_config11 = None

        # Input 12
        in12 = None
        dtype12 = None
        mem_config12 = None

        # Layer 0: input
        in0_t = torch2tt_tensor(in0, self.device, tt_memory_config=mem_config0, tt_dtype=dtype0)
        in1_t = torch2tt_tensor(in1, self.device, tt_memory_config=mem_config1, tt_dtype=dtype1)
        in2_t = torch2tt_tensor(in2, self.device, tt_memory_config=mem_config2, tt_dtype=dtype2)
        in3_t = torch2tt_tensor(in3, self.device, tt_memory_config=mem_config3, tt_dtype=dtype3)
        in4_t = torch2tt_tensor(in4, self.device, tt_memory_config=mem_config4, tt_dtype=dtype4)
        in5_t = torch2tt_tensor(in5, self.device, tt_memory_config=mem_config5, tt_dtype=dtype5)
        in6_t = torch2tt_tensor(in6, self.device, tt_memory_config=mem_config6, tt_dtype=dtype6)
        in7_t = torch2tt_tensor(in7, self.device, tt_memory_config=mem_config7, tt_dtype=dtype7)
        in8_t = torch2tt_tensor(in8, self.device, tt_memory_config=mem_config8, tt_dtype=dtype8)
        in9_t = torch2tt_tensor(in9, self.device, tt_memory_config=mem_config9, tt_dtype=dtype9)
        in10_t = torch2tt_tensor(in10, self.device, tt_memory_config=mem_config10, tt_dtype=dtype10)
        in11_t = torch2tt_tensor(in11, self.device, tt_memory_config=mem_config11, tt_dtype=dtype11)
        in12_t = torch2tt_tensor(in12, self.device, tt_memory_config=mem_config12, tt_dtype=dtype12)

        # Layer 1: matmul
        mm1 = ttnn.matmul(in0_t, in1_t)
        mm2 = ttnn.matmul(in1_t, in2_t)
        mm3 = ttnn.matmul(in2_t, in3_t)

        # Layer 2: matmul
        mm4 = ttnn.matmul(in4_t, mm1)
        mm5 = ttnn.matmul(mm2, in10_t)
        mm6 = ttnn.matmul(mm3, in7_t)

        # Layer 3: matmul
        mm7 = ttnn.matmul(in5_t, mm4)
        mm8 = ttnn.matmul(mm5, in11_t)
        mm9 = ttnn.matmul(mm6, in8_t)

        # Layer 4: matmul
        mm10 = ttnn.matmul(in6_t, mm7)
        mm11 = ttnn.matmul(mm8, in12_t)
        mm12 = ttnn.matmul(mm9, in9_t)

        # Layer 5: matmul
        mm13 = ttnn.matmul(mm10, mm11)
        mm14 = ttnn.matmul(mm11, mm12)

        # Layer 6: matmul
        mm15 = ttnn.matmul(mm13, mm14)
        out = tt2torch_tensor(mm15)

        return out

"""
SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

SPDX-License-Identifier: Apache-2.0
"""

import tt_lib
import torch.nn as nn


class TtBaddbmm(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.mem_config = tt_lib.tensor.MemoryConfig(True, tt_lib.tensor.BufferType.L1)

    def forward(
        self,
        input: tt_lib.tensor.Tensor,
        batch1: tt_lib.tensor.Tensor,
        batch2: tt_lib.tensor.Tensor,
        beta: float = 1.0,
        alpha: float = 1.0,
    ) -> tt_lib.tensor.Tensor:
        if beta != 1.0:
            input = tt_lib.tensor.mul(beta, input, output_mem_config=self.mem_config)

        tmp = tt_lib.tensor.bmm(batch1, batch2, output_mem_config=self.mem_config)

        if alpha != 1.0:
            tmp = tt_lib.tensor.mul(alpha, tmp, output_mem_config=self.mem_config)

        result = tt_lib.tensor.add(input, tmp, output_mem_config=self.mem_config)

        return result

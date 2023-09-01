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

    def forward(
        self,
        input: tt_lib.tensor.Tensor,
        batch1: tt_lib.tensor.Tensor,
        batch2: tt_lib.tensor.Tensor,
        beta: float = 1.0,
        alpha: float = 1.0,
    ) -> tt_lib.tensor.Tensor:
        if beta != 1.0:
            input = tt_lib.tensor.mul(beta, input)

        tmp = tt_lib.tensor.bmm(batch1, batch2)

        if alpha != 1.0:
            tmp = tt_lib.tensor.mul(alpha, tmp)

        result = tt_lib.tensor.add(input, tmp)

        return result

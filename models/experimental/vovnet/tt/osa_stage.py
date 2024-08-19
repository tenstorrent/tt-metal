# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

from tt_lib import fallback_ops
from models.experimental.vovnet.tt.osa_block import TtOsaBlock
import ttnn


class TtOsaStage(nn.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        block_per_stage: int = 1,
        layer_per_block: int = 3,
        groups: int = 64,
        downsample=True,
        residual=True,
        depthwise=True,
        base_address=None,
        device=None,
        state_dict=None,
    ):
        super(TtOsaStage, self).__init__()
        self.device = device
        self.base_address = f"{base_address}.blocks.0"
        self.state_dict = state_dict
        if downsample:
            self.pool = fallback_ops.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        else:
            self.pool = None

        blocks = []
        for i in range(block_per_stage):
            last_block = i == block_per_stage - 1
            blocks += [
                TtOsaBlock(
                    in_chs=in_chs,
                    mid_chs=mid_chs,
                    out_chs=out_chs,
                    layer_per_block=3,
                    residual=residual,
                    depthwise=depthwise,
                    base_address=self.base_address,
                    state_dict=self.state_dict,
                    device=self.device,
                )
            ]
            in_chs = out_chs
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self.pool is not None:
            x = self.pool(x)
        x = self.blocks(x)
        return x

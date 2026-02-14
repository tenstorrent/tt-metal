# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Sequence, Tuple
from torch import Tensor
from torch import nn as nn
import warnings


class SECOND(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: Sequence[int] = [128, 128, 256],
        layer_nums: Sequence[int] = [3, 5, 5],
        layer_strides: Sequence[int] = [2, 2, 2],
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        conv_cfg=dict(type="Conv2d", bias=False),
        init_cfg=None,
        pretrained: Optional[str] = None,
    ) -> None:
        super(SECOND, self).__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                nn.Conv2d(in_filters[i], out_channels[i], 3, bias=False, stride=layer_strides[i], padding=1),
                nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01) if norm_cfg["type"] == "BN2d" else None,
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1))
                block.append(
                    nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01) if norm_cfg["type"] == "BN2d" else None
                )
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        assert not (init_cfg and pretrained), "init_cfg and pretrained cannot be setting at the same time"
        if isinstance(pretrained, str):
            warnings.warn("DeprecationWarning: pretrained is a deprecated, " 'please use "init_cfg" instead')
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        else:
            self.init_cfg = dict(type="Kaiming", layer="Conv2d")

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)

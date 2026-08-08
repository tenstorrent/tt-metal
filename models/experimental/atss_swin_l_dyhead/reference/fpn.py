# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Standalone Feature Pyramid Network for ATSS detection.
# Extracted from MMDetection v3.3.0 (mmdet.models.necks.fpn)
# and converted to dependency-free PyTorch.

from __future__ import annotations

from typing import List, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FPN(nn.Module):
    """Feature Pyramid Network.

    Takes multi-scale backbone features and produces a set of feature maps
    with a unified channel dimension, plus optional extra levels via
    stride-2 convolutions.

    Args:
        in_channels: Number of input channels per scale.
        out_channels: Number of output channels (used at each scale).
        num_outs: Number of output scales.
        start_level: Index of the start input backbone level.
        end_level: Index of the end input backbone level (exclusive).
        add_extra_convs: Source of extra conv inputs ('on_input', 'on_lateral', 'on_output').
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = False,
        relu_before_extra_convs: bool = False,
    ):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1

        self.start_level = start_level
        self.end_level = end_level

        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ("on_input", "on_lateral", "on_output")
            self.add_extra_convs = add_extra_convs
        elif add_extra_convs:
            self.add_extra_convs = "on_input"
        else:
            self.add_extra_convs = False

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2d(in_channels[i], out_channels, 1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == "on_input":
                    in_ch = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_ch = out_channels
                extra_fpn_conv = nn.Conv2d(in_ch, out_channels, 3, stride=2, padding=1)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        assert len(inputs) == len(self.in_channels)

        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, mode="nearest")

        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == "on_input":
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == "on_lateral":
                    extra_source = laterals[-1]
                elif self.add_extra_convs == "on_output":
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)


def build_fpn_for_atss() -> FPN:
    """Instantiate FPN matching the ATSS config.

    in_channels=[384, 768, 1536] from Swin-L stages (1, 2, 3).
    num_outs=5 with add_extra_convs='on_output' → P3, P4, P5, P6, P7.
    """
    return FPN(
        in_channels=[384, 768, 1536],
        out_channels=256,
        num_outs=5,
        start_level=0,
        add_extra_convs="on_output",
    )

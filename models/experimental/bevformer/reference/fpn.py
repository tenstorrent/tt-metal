# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Feature Pyramid Network (FPN) neck in PyTorch.

This module implements the image neck of BEVFormer-base, which turns the backbone's C3,
C4 and C5 (512, 1024 and 2048 channels) into four 256-channel levels for the encoder's
spatial cross-attention. It is the reference the TTNN neck in ``tt/tt_fpn.py`` is checked
against. The ``img_neck`` weights of the BEVFormer-base checkpoint load into it with no
key changes.

The FPN performs:
1. A 1x1 lateral conv on every input level
2. A top-down path adding each coarser lateral, upsampled, to the next finer one
3. A 3x3 output conv on every level
4. Extra levels from stride-2 3x3 convs, on the last output for BEVFormer-base

BEVFormer-base's convs carry no norm or activation, so mmcv's ConvModule is reduced to a
bare conv.

Adapted from the UniAD port in ``models/experimental/uniad/reference/fpn.py``, which is
based on the mmdetection version BEVFormer is built on:
https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/necks/fpn.py
https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/cnn/bricks/conv_module.py

BEVFormer-base neck configuration:
https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_base.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union


class ConvModule(nn.Module):
    """mmcv's ConvModule reduced to the case this FPN uses: a bare conv, no norm or activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        conv_cfg: Optional[Dict] = None,
        norm_cfg: Optional[Dict] = None,
        act_cfg: Optional[Dict] = None,
        inplace: bool = True,
    ):
        super().__init__()
        assert conv_cfg is None, f"only a plain Conv2d is supported, got conv_cfg={conv_cfg}"
        assert norm_cfg is None, f"norm layers are not supported, got norm_cfg={norm_cfg}"
        assert act_cfg is None, f"activations are not supported, got act_cfg={act_cfg}"
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FPN(nn.Module):
    """
    Feature Pyramid Network over the backbone levels ``start_level`` .. ``end_level``.

    Args:
        in_channels (list[int]): Channels of each backbone level.
        out_channels (int): Channels of every output level.
        num_outs (int): Number of output levels; those beyond the backbone levels are extra.
        add_extra_convs (bool | str): False makes the extra levels by max pooling the last
            output. Otherwise a stride-2 3x3 conv makes each one, the first reading
            ``"on_input"`` (the last backbone input, also what True means), ``"on_lateral"``
            (the last lateral) or ``"on_output"`` (the last output).
        relu_before_extra_convs (bool): Apply ReLU before every extra conv after the first.

    ``init_cfg`` is accepted for config compatibility only: weights come from a checkpoint
    or from the tests' initializers.

    Returns:
        tuple[torch.Tensor]: ``num_outs`` (N, out_channels, H, W) tensors, finest first.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        add_extra_convs=False,
        relu_before_extra_convs=False,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        upsample_cfg=dict(mode="nearest"),
        init_cfg=dict(type="Xavier", layer="Conv2d", distribution="uniform"),
    ):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ("on_input", "on_lateral", "on_output")
        elif add_extra_convs:  # True
            self.add_extra_convs = "on_input"

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == "on_input":
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False,
                )
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if "scale_factor" in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
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

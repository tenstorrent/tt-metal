# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Standalone ATSS detection head (inference-only).
# Extracted from MMDetection v3.3.0 (mmdet.models.dense_heads.atss_head)
# and converted to dependency-free PyTorch.
#
# For the Swin-L + DyHead config, stacked_convs=0 so the head
# is just three 1x1 convolutions (cls, reg, centerness) per FPN level.

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class Scale(nn.Module):
    """Learnable per-level scale factor (initialized to 1.0)."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale


class ATSSHead(nn.Module):
    """ATSS detection head (inference-only).

    With stacked_convs=0 (DyHead provides the stacked convolutions),
    this is simply three 1x1 conv branches per FPN level:
    - Classification: in_channels → num_classes
    - Regression: in_channels → 4
    - Centerness: in_channels → 1

    Plus a learnable Scale per level for the regression branch.

    Args:
        num_classes: Number of object categories (COCO=80).
        in_channels: Number of input channels from the neck.
        feat_channels: Number of intermediate feature channels.
        stacked_convs: Number of stacked conv layers (0 when using DyHead).
        pred_kernel_size: Kernel size for prediction convolutions.
        num_anchors: Number of anchors per spatial position.
        num_levels: Number of FPN levels.
    """

    def __init__(
        self,
        num_classes: int = 80,
        in_channels: int = 256,
        feat_channels: int = 256,
        stacked_convs: int = 0,
        pred_kernel_size: int = 1,
        num_anchors: int = 1,
        num_levels: int = 5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.pred_kernel_size = pred_kernel_size
        self.num_anchors = num_anchors

        self._init_layers(num_levels)

    def _init_layers(self, num_levels: int) -> None:
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                nn.Sequential(
                    nn.Conv2d(chn, self.feat_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(32, self.feat_channels),
                    nn.ReLU(inplace=True),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    nn.Conv2d(chn, self.feat_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(32, self.feat_channels),
                    nn.ReLU(inplace=True),
                )
            )

        pred_pad = self.pred_kernel_size // 2
        self.atss_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.num_classes,
            self.pred_kernel_size,
            padding=pred_pad,
        )
        self.atss_reg = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * 4,
            self.pred_kernel_size,
            padding=pred_pad,
        )
        self.atss_centerness = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * 1,
            self.pred_kernel_size,
            padding=pred_pad,
        )
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(num_levels)])

    def forward_single(self, x: Tensor, scale: Scale) -> Tuple[Tensor, Tensor, Tensor]:
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        cls_score = self.atss_cls(cls_feat)
        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        centerness = self.atss_centerness(reg_feat)
        return cls_score, bbox_pred, centerness

    def forward(self, feats: Tuple[Tensor, ...]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        cls_scores = []
        bbox_preds = []
        centernesses = []
        for feat, scale in zip(feats, self.scales):
            cls_score, bbox_pred, centerness = self.forward_single(feat, scale)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            centernesses.append(centerness)
        return cls_scores, bbox_preds, centernesses


def build_atss_head() -> ATSSHead:
    """Instantiate ATSS head matching the Swin-L + DyHead config."""
    return ATSSHead(
        num_classes=80,
        in_channels=256,
        feat_channels=256,
        stacked_convs=0,
        pred_kernel_size=1,
        num_anchors=1,
        num_levels=5,
    )

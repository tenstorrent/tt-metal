# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Standalone DyHead (Dynamic Head) for ATSS detection.
# Extracted from MMDetection v3.3.0 (mmdet.models.necks.dyhead)
# and converted to dependency-free PyTorch.
#
# DyHead uses modulated deformable convolution (DCNv2) which requires
# torchvision.ops.deform_conv2d.

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torchvision.ops import deform_conv2d

    HAS_DEFORM_CONV = True
except ImportError:
    HAS_DEFORM_CONV = False


class HSigmoid(nn.Module):
    """Hard Sigmoid activation: clamp(x + bias, 0, 6) / divisor."""

    def __init__(self, bias: float = 3.0, divisor: float = 6.0):
        super().__init__()
        self.bias = bias
        self.divisor = divisor

    def forward(self, x: Tensor) -> Tensor:
        return (x + self.bias).clamp(min=0, max=6) / self.divisor


class DyReLU(nn.Module):
    """Dynamic ReLU — task-aware attention module from DyHead.

    Learns per-channel piecewise-linear activation parameters via
    squeeze-and-excitation style pooling.
    """

    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        self.channels = channels
        self.expansion = 4  # a1, b1, a2, b2

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, 1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels // ratio, channels * self.expansion, 1),
            HSigmoid(bias=3.0, divisor=6.0),
        )

    def forward(self, x: Tensor) -> Tensor:
        coeffs = self.global_avgpool(x)
        coeffs = self.conv1(coeffs)
        coeffs = self.conv2(coeffs) - 0.5
        a1, b1, a2, b2 = torch.split(coeffs, self.channels, dim=1)
        a1 = a1 * 2.0 + 1.0
        a2 = a2 * 2.0
        return torch.max(x * a1 + b1, x * a2 + b2)


class ModulatedDeformConv2dFunction(nn.Module):
    """Modulated Deformable Convolution using torchvision.ops.deform_conv2d.

    Wraps the functional API with learnable weight and optional bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        nn.init.kaiming_uniform_(self.weight, a=1)

    def forward(self, x: Tensor, offset: Tensor, mask: Tensor) -> Tensor:
        if not HAS_DEFORM_CONV:
            raise RuntimeError(
                "torchvision.ops.deform_conv2d is required for DyHead. "
                "Install torchvision with: pip install torchvision"
            )
        return deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            mask=mask,
        )


class DyDCNv2(nn.Module):
    """Modulated Deformable Conv2d with GroupNorm used in DyHead."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        num_groups: int = 16,
    ):
        super().__init__()
        self.conv = ModulatedDeformConv2dFunction(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x: Tensor, offset: Tensor, mask: Tensor) -> Tensor:
        x = self.conv(x.contiguous(), offset, mask)
        x = self.norm(x)
        return x


class DyHeadBlock(nn.Module):
    """Single DyHead block with spatial, scale, and task-aware attention.

    For each FPN level, the block:
    1. Computes DCNv2 offset/mask from the current level
    2. Applies spatial-aware attention via DCNv2 on mid/low/high levels
    3. Applies scale-aware attention via adaptive avg pooling
    4. Applies task-aware attention via DyReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        zero_init_offset: bool = True,
    ):
        super().__init__()
        self.zero_init_offset = zero_init_offset
        self.offset_and_mask_dim = 3 * 3 * 3  # (offset_x, offset_y, mask) * 3 * 3 kernel
        self.offset_dim = 2 * 3 * 3

        self.spatial_conv_high = DyDCNv2(in_channels, out_channels)
        self.spatial_conv_mid = DyDCNv2(in_channels, out_channels)
        self.spatial_conv_low = DyDCNv2(in_channels, out_channels, stride=2)
        self.spatial_conv_offset = nn.Conv2d(in_channels, self.offset_and_mask_dim, 3, padding=1)

        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, 1, 1),
            nn.ReLU(inplace=True),
            HSigmoid(bias=3.0, divisor=6.0),
        )
        self.task_attn_module = DyReLU(out_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if self.zero_init_offset:
            nn.init.constant_(self.spatial_conv_offset.weight, 0)
            nn.init.constant_(self.spatial_conv_offset.bias, 0)

    @staticmethod
    def _resize_offset_mask(offset: Tensor, mask: Tensor, target_size: Tuple[int, int]) -> Tuple[Tensor, Tensor]:
        """Resize offset and mask to match the expected conv output spatial dims.

        This is needed because the offset is computed from the current level
        but applied to adjacent levels with different spatial resolutions.
        mmcv's ModulatedDeformConv2d handles mismatched sizes internally,
        but torchvision's deform_conv2d requires exact match.
        """
        if offset.shape[-2:] != target_size:
            offset = F.interpolate(offset, size=target_size, mode="bilinear", align_corners=True)
            # Scale offset values proportionally to the spatial resize
            h_scale = target_size[0] / offset.shape[-2] if offset.shape[-2] != target_size[0] else 1.0
            w_scale = target_size[1] / offset.shape[-1] if offset.shape[-1] != target_size[1] else 1.0
            mask = F.interpolate(mask, size=target_size, mode="bilinear", align_corners=True)
        return offset, mask

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        outs = []
        for level in range(len(x)):
            offset_and_mask = self.spatial_conv_offset(x[level])
            offset = offset_and_mask[:, : self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim :, :, :].sigmoid()

            mid_feat = self.spatial_conv_mid(x[level], offset, mask)
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1

            if level > 0:
                # Low-level feature (previous level) is larger; stride=2 conv
                # produces output matching current level dims → offset matches
                low_offset, low_mask = offset, mask
                low_input = x[level - 1]
                # stride-2 conv: output_size = input_size / 2 → should match offset size
                expected_low_out = (low_input.shape[2] // 2, low_input.shape[3] // 2)
                if expected_low_out != offset.shape[-2:]:
                    low_offset, low_mask = self._resize_offset_mask(offset, mask, expected_low_out)
                low_feat = self.spatial_conv_low(low_input, low_offset, low_mask)
                sum_feat = sum_feat + low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1

            if level < len(x) - 1:
                # High-level feature (next level) is smaller; stride=1 conv
                # produces output at next level's resolution → resize offset
                high_input = x[level + 1]
                high_target = (high_input.shape[2], high_input.shape[3])
                high_offset, high_mask = self._resize_offset_mask(offset, mask, high_target)
                high_feat = F.interpolate(
                    self.spatial_conv_high(high_input, high_offset, high_mask),
                    size=x[level].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                sum_feat = sum_feat + high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1

            outs.append(self.task_attn_module(sum_feat / summed_levels))
        return outs


class DyHead(nn.Module):
    """DyHead neck: multiple DyHead blocks in sequence.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_blocks: Number of DyHead blocks.
        zero_init_offset: Whether to zero-init DCNv2 offsets.
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        num_blocks: int = 6,
        zero_init_offset: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks

        blocks = []
        for i in range(num_blocks):
            in_ch = in_channels if i == 0 else out_channels
            blocks.append(DyHeadBlock(in_ch, out_channels, zero_init_offset=zero_init_offset))
        self.dyhead_blocks = nn.Sequential(*blocks)

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        assert isinstance(inputs, (tuple, list))
        outs = self.dyhead_blocks(list(inputs))
        return outs


def build_dyhead_for_atss() -> DyHead:
    """Instantiate DyHead matching the ATSS config."""
    return DyHead(
        in_channels=256,
        out_channels=256,
        num_blocks=6,
        zero_init_offset=False,
    )

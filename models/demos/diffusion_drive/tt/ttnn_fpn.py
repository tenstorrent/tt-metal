# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TTNN implementation of TransfuserBackbone._top_down (3-level FPN).

Three plain Conv2d layers (no BatchNorm) run on TTNN.  Bilinear upsampling
stays in PyTorch because ttnn.upsample nearest-only diverges from bilinear at
these scales; the two upsample steps are cheap relative to the conv work.

FPN topology:
    lidar_feats (B, 512, 8, 8)
        → c5_conv   1×1  512→64   + ReLU          → (B, 64, 8, 8)   [TTNN]
        → upsample  2×   bilinear                  → (B, 64, 16, 16) [PyTorch]
        → up_conv5  3×3  64→64    + ReLU          → (B, 64, 16, 16) [TTNN]
        → upsample2 to (64,64)  bilinear           → (B, 64, 64, 64) [PyTorch]
        → up_conv4  3×3  64→64    + ReLU          → (B, 64, 64, 64) [TTNN]
        = bev_upscale

Public class:
    TtnnFPN  — drop-in for TransfuserBackbone._top_down
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn
from models.demos.diffusion_drive.tt.ttnn_backbone import _from_ttnn_tile, _to_ttnn_tile
from models.demos.diffusion_drive.tt.ttnn_resnet34 import _ttnn_conv2d


def _prep(conv: nn.Conv2d):
    """Extract bfloat16 weight/bias from a plain Conv2d (no BN fold needed)."""
    w = conv.weight.detach().to(torch.bfloat16)
    b = (
        conv.bias.detach().to(torch.bfloat16)
        if conv.bias is not None
        else torch.zeros(conv.out_channels, dtype=torch.bfloat16)
    )
    return w, b


class TtnnFPN:
    """
    TTNN top-down FPN replacing TransfuserBackbone._top_down.

    Parameters
    ----------
    ref_backbone : TransfuserBackbone
        Provides c5_conv, up_conv5, up_conv4, upsample, upsample2.
    device : ttnn.Device
    """

    def __init__(self, ref_backbone, device: ttnn.Device) -> None:
        self._device = device

        # Pre-extract weights (no BN fold — these convs have no BN)
        self._c5_w, self._c5_b = _prep(ref_backbone.c5_conv)  # 1×1  512→64
        self._up5_w, self._up5_b = _prep(ref_backbone.up_conv5)  # 3×3  64→64
        self._up4_w, self._up4_b = _prep(ref_backbone.up_conv4)  # 3×3  64→64

        # Upsample parameters (used with PyTorch F.interpolate)
        self._scale = float(ref_backbone.upsample.scale_factor)
        self._size2 = ref_backbone.upsample2.size  # tuple (H, W) = (64, 64)

    # ------------------------------------------------------------------
    def _conv_relu(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        ksize: int,
        stride: int,
        pad: int,
    ) -> torch.Tensor:
        """Run one TTNN conv2d + relu; returns a PyTorch NCHW float32 tensor."""
        B, C_in, H, W = x.shape
        C_out = w.shape[0]
        x_tt = _to_ttnn_tile(x, B, H, W, C_in, self._device)
        out_tt, H_out, W_out = _ttnn_conv2d(self._device, x_tt, w, b, B, H, W, C_in, C_out, ksize, stride, pad)
        if out_tt.is_sharded():
            out_tt = ttnn.sharded_to_interleaved(out_tt, ttnn.DRAM_MEMORY_CONFIG)
        out_tt = ttnn.relu(out_tt)
        return _from_ttnn_tile(out_tt, B, H_out, W_out, C_out)

    # ------------------------------------------------------------------
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 512, 8, 8) float32 — deepest lidar feature from backbone.
        Returns:
            (B, 64, 64, 64) float32 — bev_upscale (P3).
        """
        # c5_conv: 1×1, 512→64, relu
        p5 = self._conv_relu(x, self._c5_w, self._c5_b, ksize=1, stride=1, pad=0)

        # bilinear 8×8 → 16×16  (PyTorch)
        p5_up = F.interpolate(p5, scale_factor=self._scale, mode="bilinear", align_corners=False)

        # up_conv5: 3×3, 64→64, relu
        p4 = self._conv_relu(p5_up, self._up5_w, self._up5_b, ksize=3, stride=1, pad=1)

        # bilinear 16×16 → 64×64 (PyTorch)
        p4_up = F.interpolate(p4, size=self._size2, mode="bilinear", align_corners=False)

        # up_conv4: 3×3, 64→64, relu
        p3 = self._conv_relu(p4_up, self._up4_w, self._up4_b, ksize=3, stride=1, pad=1)

        return p3

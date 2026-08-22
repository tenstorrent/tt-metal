# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TTNN implementation of TransfuserBackbone._top_down (3-level FPN).

Three plain Conv2d layers (no BatchNorm) AND both bilinear upsamples run on
TTNN.  ttnn.upsample(mode="bilinear") matches torch align_corners=False to
PCC ≥ 0.99999 for the FPN's integer ×2 / ×4 scales, so the previous PyTorch
F.interpolate fallback has been removed.

FPN topology:
    lidar_feats (B, 512, 8, 8)
        → c5_conv   1×1  512→64   + ReLU          → (B, 64, 8, 8)   [TTNN]
        → upsample  2×   bilinear                  → (B, 64, 16, 16) [TTNN]
        → up_conv5  3×3  64→64    + ReLU          → (B, 64, 16, 16) [TTNN]
        → upsample2 to (64,64)  bilinear           → (B, 64, 64, 64) [TTNN]
        → up_conv4  3×3  64→64    + ReLU          → (B, 64, 64, 64) [TTNN]
        = bev_upscale

Public class:
    TtnnFPN  — drop-in for TransfuserBackbone._top_down
"""

from __future__ import annotations

import torch
import torch.nn as nn

import ttnn
from models.demos.diffusion_drive.tt.ttnn_backbone import _from_ttnn_tile, _to_ttnn_tile
from models.demos.diffusion_drive.tt.ttnn_resnet34 import _RELU_ACT, _ttnn_conv2d, prep_conv_weights


def _prep(conv: nn.Conv2d):
    """Extract bfloat16 weight/bias from a plain Conv2d (no BN fold needed) and
    pre-convert to TTNN host tensors once."""
    w = conv.weight.detach().to(torch.bfloat16)
    b = (
        conv.bias.detach().to(torch.bfloat16)
        if conv.bias is not None
        else torch.zeros(conv.out_channels, dtype=torch.bfloat16)
    )
    return prep_conv_weights(w, b)


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
        # Output-channel counts taken from the original convs — after the first
        # forward the cached device weights conv2d returns no longer expose C_out
        # via .shape[0], so we must not read it off the weight tensor.
        self._c5_cout = ref_backbone.c5_conv.out_channels
        self._up5_cout = ref_backbone.up_conv5.out_channels
        self._up4_cout = ref_backbone.up_conv4.out_channels

        # Upsample parameters (used with PyTorch F.interpolate)
        self._scale = float(ref_backbone.upsample.scale_factor)
        self._size2 = ref_backbone.upsample2.size  # tuple (H, W) = (64, 64)

    # ------------------------------------------------------------------
    def _upsample_bilinear(self, x: torch.Tensor, scale_h: int, scale_w: int) -> torch.Tensor:
        """Bilinear upsample on TTNN device (replaces F.interpolate fallback).

        ttnn.upsample(mode="bilinear") matches torch align_corners=False to
        PCC ≥ 0.99999 for integer scales (verified for the FPN's ×2 and ×4
        steps).  Input arrives as torch NCHW; round-trips through NHWC
        ROW_MAJOR on device and returns torch NCHW.
        """
        B, C, H, W = x.shape
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        x_tt = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self._device)
        out_tt = ttnn.upsample(x_tt, [int(scale_h), int(scale_w)], mode="bilinear")
        out = ttnn.to_torch(out_tt)  # (B, H*scale_h, W*scale_w, C)
        return out.reshape(B, H * scale_h, W * scale_w, C).permute(0, 3, 1, 2).float()

    # ------------------------------------------------------------------
    def _conv_relu(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        ksize: int,
        stride: int,
        pad: int,
        c_out: int,
    ):
        """Run one TTNN conv2d + relu.

        Returns ``(out_nchw_float32, prepared_w, prepared_b)``.  The prepared
        weight/bias are the device-resident tensors ``ttnn.conv2d`` returns; the
        caller reassigns them so the next forward skips the host re-upload (a
        small perf win and a prerequisite for trace capture, which forbids
        host→device writes).  ``c_out`` is passed in rather than read from
        ``w.shape[0]`` because the cached device weight no longer exposes it.
        """
        B, C_in, H, W = x.shape
        x_tt = _to_ttnn_tile(x, B, H, W, C_in, self._device)
        out_tt, H_out, W_out, prep_w, prep_b = _ttnn_conv2d(
            self._device, x_tt, w, b, B, H, W, C_in, c_out, ksize, stride, pad, activation=_RELU_ACT
        )
        if out_tt.is_sharded():
            out_tt = ttnn.sharded_to_interleaved(out_tt, ttnn.DRAM_MEMORY_CONFIG)
        return _from_ttnn_tile(out_tt, B, H_out, W_out, c_out), prep_w, prep_b

    # ------------------------------------------------------------------
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 512, 8, 8) float32 — deepest lidar feature from backbone.
        Returns:
            (B, 64, 64, 64) float32 — bev_upscale (P3).
        """
        # c5_conv: 1×1, 512→64, relu
        p5, self._c5_w, self._c5_b = self._conv_relu(
            x, self._c5_w, self._c5_b, ksize=1, stride=1, pad=0, c_out=self._c5_cout
        )

        # bilinear 8×8 → 16×16  (TTNN ttnn.upsample, integer scale)
        s = int(round(self._scale))
        p5_up = self._upsample_bilinear(p5, s, s)

        # up_conv5: 3×3, 64→64, relu
        p4, self._up5_w, self._up5_b = self._conv_relu(
            p5_up, self._up5_w, self._up5_b, ksize=3, stride=1, pad=1, c_out=self._up5_cout
        )

        # bilinear 16×16 → 64×64 (TTNN ttnn.upsample, derived integer scale)
        scale_h = self._size2[0] // p4.shape[2]
        scale_w = self._size2[1] // p4.shape[3]
        p4_up = self._upsample_bilinear(p4, scale_h, scale_w)

        # up_conv4: 3×3, 64→64, relu
        p3, self._up4_w, self._up4_b = self._conv_relu(
            p4_up, self._up4_w, self._up4_b, ksize=3, stride=1, pad=1, c_out=self._up4_cout
        )

        return p3

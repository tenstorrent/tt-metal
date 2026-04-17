# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""On-device port of the DROID-SLAM BasicEncoder (fnet / cnet)."""

from __future__ import annotations

import torch
import ttnn

from models.experimental.droid_slam.tt.ttnn_layers import (
    RELU,
    TtConv2d,
    TtInstanceNorm2d,
)


class _TtResidualBlock:
    """Residual block whose every op runs on device.

    Supports both `norm_fn='instance'` (InstanceNorm2d) and
    `norm_fn='none'` (identity norm). ReLU is fused into the conv where
    it immediately follows; the post-residual-add ReLU is fused via
    `ttnn.add(..., activations=[RELU])`.
    """

    def __init__(self, block: torch.nn.Module, norm_fn: str, device):
        self.norm_fn = norm_fn
        # conv1 → norm1 → ReLU. When norm is identity we can fuse ReLU
        # into the conv itself; when it's instance norm we must run the
        # norm first and then relu the result.
        fuse_relu_into_conv = norm_fn == "none"
        self.conv1 = TtConv2d(
            block.conv1, activation=RELU if fuse_relu_into_conv else None
        )
        # Torch ResidualBlock.forward applies ReLU AFTER norm2 but
        # BEFORE the residual add — same fusion rule as conv1.
        self.conv2 = TtConv2d(
            block.conv2, activation=RELU if fuse_relu_into_conv else None
        )

        self.norm1 = (
            TtInstanceNorm2d(block.conv1.out_channels, device)
            if norm_fn == "instance"
            else None
        )
        self.norm2 = (
            TtInstanceNorm2d(block.conv2.out_channels, device)
            if norm_fn == "instance"
            else None
        )

        if block.downsample is not None:
            ds_conv = block.downsample[0]  # Conv2d(1x1, stride)
            self.downsample = TtConv2d(ds_conv, activation=None)
            # downsample's norm mirrors the layout of the block.
            self.downsample_norm = (
                TtInstanceNorm2d(ds_conv.out_channels, device)
                if norm_fn == "instance"
                else None
            )
        else:
            self.downsample = None
            self.downsample_norm = None

        self.batch_size = None  # set per-forward — ttnn.conv2d needs it explicitly.

    @staticmethod
    def _maybe_norm_relu(x, norm, batch_size, spatial):
        if norm is None:
            # conv already fused ReLU — just return.
            return x
        x = norm(x, batch_size=batch_size, spatial=spatial)
        return ttnn.relu(x)

    def __call__(self, x, device, batch_size, h, w):
        residual = x
        res_h, res_w = h, w

        x, h1, w1 = self.conv1(x, device, batch_size, h, w)
        x = self._maybe_norm_relu(x, self.norm1, batch_size, h1 * w1)

        x, h2, w2 = self.conv2(x, device, batch_size, h1, w1)

        # norm2 + ReLU run *before* the residual add (matches torch).
        if self.norm2 is not None:
            x = self.norm2(x, batch_size=batch_size, spatial=h2 * w2)
            x = ttnn.relu(x)

        if self.downsample is not None:
            residual, res_h, res_w = self.downsample(
                residual, device, batch_size, res_h, res_w
            )
            if self.downsample_norm is not None:
                residual = self.downsample_norm(
                    residual, batch_size=batch_size, spatial=res_h * res_w
                )

        # Move both operands to a common interleaved DRAM layout before
        # the add — conv outputs can land in differently-sharded L1
        # buffers which makes ttnn.add silently produce stale results.
        x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        residual = ttnn.sharded_to_interleaved(residual, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.add(x, residual, activations=[RELU])
        return out, h2, w2


class TtBasicEncoder:
    """DROID-SLAM BasicEncoder on-device.

    The caller is responsible for reshaping `(B, N, 3, H, W)` torch
    input into `(B*N, 3, H, W)` before upload; we treat that flattened
    batch as ttnn `batch_size`.
    """

    def __init__(self, ref_encoder, device, skip_conv2=False):
        self.device = device
        self.norm_fn = ref_encoder.norm_fn
        self.skip_conv2 = skip_conv2
        self.conv1 = TtConv2d(
            ref_encoder.conv1,
            activation=RELU if self.norm_fn == "none" else None,
        )
        self.norm1 = (
            TtInstanceNorm2d(ref_encoder.conv1.out_channels, device)
            if self.norm_fn == "instance"
            else None
        )
        self.layer1 = [
            _TtResidualBlock(ref_encoder.layer1[0], self.norm_fn, device),
            _TtResidualBlock(ref_encoder.layer1[1], self.norm_fn, device),
        ]
        self.layer2 = [
            _TtResidualBlock(ref_encoder.layer2[0], self.norm_fn, device),
            _TtResidualBlock(ref_encoder.layer2[1], self.norm_fn, device),
        ]
        self.layer3 = [
            _TtResidualBlock(ref_encoder.layer3[0], self.norm_fn, device),
            _TtResidualBlock(ref_encoder.layer3[1], self.norm_fn, device),
        ]
        self.conv2 = (
            None if skip_conv2 else TtConv2d(ref_encoder.conv2, activation=None)
        )
        self.out_channels = ref_encoder.conv2.out_channels

    def __call__(self, x_tile, batch_size, h, w):
        """x_tile is an on-device tile NHWC tensor; returns (tile, h_out, w_out)."""
        x, h, w = self.conv1(x_tile, self.device, batch_size, h, w)
        if self.norm1 is not None:
            x = self.norm1(x, batch_size=batch_size, spatial=h * w)
            x = ttnn.relu(x)
        for block in self.layer1:
            x, h, w = block(x, self.device, batch_size, h, w)
        for block in self.layer2:
            x, h, w = block(x, self.device, batch_size, h, w)
        for block in self.layer3:
            x, h, w = block(x, self.device, batch_size, h, w)
        if self.conv2 is None:
            return x, h, w
        x, h, w = self.conv2(x, self.device, batch_size, h, w)
        return x, h, w

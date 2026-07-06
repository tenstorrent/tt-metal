# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `resnet_block2_d` (diffusers ``ResnetBlock2D``) for
meituan-longcat/LongCat-Image, submodule ``vae.encoder.down_blocks.0.resnets.0``.

``ResnetBlock2D.forward(input_tensor[B,C,H,W], temb)``::

    h = conv1(silu(norm1(input_tensor)))
    h = conv2(silu(norm2(h)))
    shortcut = conv_shortcut(input_tensor) if conv_shortcut is not None else input_tensor
    return (shortcut + h) / output_scale_factor        # VAE: output_scale_factor == 1

`temb` is unused: VAE resnets have ``temb_channels=None`` -> ``time_emb_proj`` is None.

This is the exact per-resnet math the graduated `down_encoder_block2_d` /
`autoencoder_k_l` VAE ports already validate, so we reuse it verbatim by
subclassing ``_DownEncoderBlock2D`` and calling its PCC-verified ``_resnet``
(fully-fp32 manual GroupNorm — ``ttnn.group_norm`` is bf16-only and is the VAE
precision limiter — with bf16 conv weights processed at full mantissa via HiFi4).
Tensor format between ops is NHWC flattened to [1,1,H*W,C]; the block I/O is NCHW
to match the torch reference.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.vision.generative.longcat_image._stubs.down_encoder_block2_d import (
    DRAM,
    F32,
    RM,
    TILE,
    _DownEncoderBlock2D,
)


class _ResnetBlock2D(_DownEncoderBlock2D):
    """One ResnetBlock2D. Reuses the graduated `_resnet`/`_conv`/`_gn_silu`
    helpers; `self.block` is the resnet module itself."""

    def __call__(self, input_tensor=None, temb=None, **_ignored):
        sample = input_tensor
        if sample is None:
            raise ValueError("resnet_block2_d stub requires `input_tensor`")
        if isinstance(sample, ttnn.Tensor):
            B, C, H, W = list(sample.shape)
            if sample.dtype != F32:
                if sample.layout != TILE:
                    sample = ttnn.to_layout(sample, TILE)
                sample = ttnn.typecast(sample, F32)
            if sample.layout != RM:
                sample = ttnn.to_layout(sample, RM)
            nhwc = ttnn.permute(sample, [0, 2, 3, 1])
            x = ttnn.reshape(nhwc, [1, 1, H * W, C])
        else:
            t = sample.to(torch.float32)
            B, C, H, W = t.shape
            x = ttnn.from_torch(
                t.permute(0, 2, 3, 1).reshape(1, 1, H * W, C).contiguous(),
                dtype=F32,
                layout=RM,
                device=self.device,
                memory_config=DRAM,
            )

        rb = self.block
        in_ch = int(rb.conv1.in_channels)
        out_ch = int(rb.conv2.out_channels)
        x = self._resnet(x, rb, in_ch, out_ch, H, W)  # [1,1,H*W,out_ch]

        nhwc = ttnn.reshape(x, [1, H, W, out_ch])
        if nhwc.layout != RM:
            nhwc = ttnn.to_layout(nhwc, RM)
        return ttnn.permute(nhwc, [0, 3, 1, 2])  # [1, out_ch, H, W]


def build(device, torch_module):
    """PCC-harness entry point: native TTNN diffusers ResnetBlock2D."""
    return _ResnetBlock2D(device, torch_module)

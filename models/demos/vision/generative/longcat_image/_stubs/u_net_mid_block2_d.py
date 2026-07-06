# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `u_net_mid_block2_d` (diffusers ``UNetMidBlock2D``) for
meituan-longcat/LongCat-Image, submodule ``vae.encoder.mid_block``.

``UNetMidBlock2D.forward(hidden_states[B,C,H,W], temb=None)``::

    hidden_states = resnets[0](hidden_states)
    hidden_states = attentions[0](hidden_states)      # spatial self-attn, 1 head, residual
    hidden_states = resnets[1](hidden_states)
    return hidden_states                              # channels preserved (C -> C)

This is EXACTLY the ``_midblock`` the graduated `encoder` / `decoder` VAE ports
already validate, so we reuse it verbatim by subclassing ``_Encoder`` and calling
its PCC-verified ``_midblock`` (fully-fp32 manual GroupNorm — ``ttnn.group_norm`` is
bf16-only and is the VAE precision limiter — bf16 conv weights at full mantissa via
HiFi4, and manual fp32 attention). `temb` is unused (VAE resnets have
``temb_channels=None``). Tensor format between ops is NHWC flattened to [1,1,H*W,C];
the block I/O is NCHW to match the torch reference.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.vision.generative.longcat_image._stubs.encoder import DRAM, F32, RM, TILE, _Encoder


class _UNetMidBlock2D(_Encoder):
    """One UNetMidBlock2D. Reuses the graduated `_midblock`/`_resnet`/`_attention`/
    `_gn_silu`/`_conv` helpers; `self.enc` is the mid-block module itself."""

    def __call__(self, hidden_states=None, temb=None, **_ignored):
        sample = hidden_states
        if sample is None:
            raise ValueError("u_net_mid_block2_d stub requires `hidden_states`")
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

        x = self._midblock(x, self.enc, C, H, W)  # [1,1,H*W,C] (C preserved)

        nhwc = ttnn.reshape(x, [1, H, W, C])
        if nhwc.layout != RM:
            nhwc = ttnn.to_layout(nhwc, RM)
        return ttnn.permute(nhwc, [0, 3, 1, 2])  # [1, C, H, W]


def build(device, torch_module):
    """PCC-harness entry point: native TTNN diffusers UNetMidBlock2D."""
    return _UNetMidBlock2D(device, torch_module)

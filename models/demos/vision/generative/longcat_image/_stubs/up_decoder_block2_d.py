# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `up_decoder_block2_d` (diffusers ``UpDecoderBlock2D``) for
meituan-longcat/LongCat-Image, submodule ``vae.decoder.up_blocks.0``.

``UpDecoderBlock2D.forward(hidden_states[B,C,H,W], temb=None, upsample_size=None)``::

    for resnet in resnets:      hidden_states = resnet(hidden_states)     # C -> out, then out -> out
    for up in upsamplers:       hidden_states = up(hidden_states)         # nearest 2x + 3x3 conv
    return hidden_states

This is EXACTLY the per-up-block math the graduated `decoder` VAE port already
validates, so we reuse its PCC-verified ``_resnet``/``_upsample``/``_gn_silu``/``_conv``
helpers by subclassing ``_Decoder`` (overriding ``__init__`` to skip the decoder-level
``conv_in`` probe). Fully-fp32 manual GroupNorm (``ttnn.group_norm`` is bf16-only and is
the VAE precision limiter) with bf16 conv weights at full mantissa via HiFi4; upsample is
nearest-2x then a 3x3 conv. `temb` is unused (VAE resnets have ``temb_channels=None``).
Tensor format between ops is NHWC flattened to [1,1,H*W,C]; the block I/O is NCHW.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.vision.generative.longcat_image._stubs.decoder import DRAM, F32, RM, TILE, _Decoder


class _UpDecoderBlock2D(_Decoder):
    """One UpDecoderBlock2D. Reuses the graduated `_resnet`/`_upsample`/`_gn_silu`/
    `_conv` helpers; `self.dec` is the up-block module itself."""

    def __init__(self, device, block):
        self.device = device
        self.dec = block.eval() if hasattr(block, "eval") else block
        self._cw = {}
        self._gn = {}
        self._lin = {}
        self._compute = None

    def __call__(self, hidden_states=None, temb=None, upsample_size=None, **_ignored):
        sample = hidden_states
        if sample is None:
            raise ValueError("up_decoder_block2_d stub requires `hidden_states`")
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

        ub = self.dec
        cc = C
        for rb in ub.resnets:
            oc = int(rb.conv2.out_channels)
            x = self._resnet(x, rb, cc, oc, H, W)
            cc = oc
        ups = getattr(ub, "upsamplers", None)
        if ups is not None and len(ups) > 0:
            for up in ups:
                x = self._upsample(x, up, cc, H, W)  # nearest 2x + 3x3 conv
                H, W = H * 2, W * 2

        nhwc = ttnn.reshape(x, [1, H, W, cc])
        if nhwc.layout != RM:
            nhwc = ttnn.to_layout(nhwc, RM)
        return ttnn.permute(nhwc, [0, 3, 1, 2])  # [1, cc, H, W]


def build(device, torch_module):
    """PCC-harness entry point: native TTNN diffusers UpDecoderBlock2D."""
    return _UpDecoderBlock2D(device, torch_module)

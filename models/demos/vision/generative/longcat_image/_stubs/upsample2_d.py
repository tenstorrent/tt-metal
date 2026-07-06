# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `upsample2_d` (diffusers ``Upsample2D``) for
meituan-longcat/LongCat-Image (VAE decoder upsample), submodule
``vae.decoder.up_blocks.0.upsamplers.0``.

``Upsample2D.forward(hidden_states)`` (use_conv=True)::

    hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
    hidden_states = conv(hidden_states)               # 3x3 stride-1, padding=1
    return hidden_states

Mirror of the graduated `downsample2_d` port (and the decoder's ``_upsample``):
fp32 activations + fp32 conv output, bf16 conv weights, HiFi4/fp32-accumulate.
Nearest 2x upsample via ``ttnn.upsample``. Tensor format between ops is NHWC
flattened to [1,1,H*W,C]; I/O is NCHW to match the torch reference.
"""

from __future__ import annotations

import torch

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
F32 = ttnn.float32
BF16 = ttnn.bfloat16
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT


class _Upsample2D:
    def __init__(self, device, up):
        self.device = device
        self.up = up.eval() if hasattr(up, "eval") else up
        self.conv = getattr(self.up, "conv", None)
        self.channels = int(self.conv.in_channels) if self.conv is not None else None
        self._cw = {}
        self._compute = None

    def _ck(self):
        if self._compute is None:
            self._compute = ttnn.init_device_compute_kernel_config(
                self.device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
            )
        return self._compute

    def _conv(self, x, tm, in_ch, out_ch, h, w, k, s, pad):
        key = id(tm)
        if key not in self._cw:
            self._cw[key] = (
                ttnn.from_torch(tm.weight.detach().to(torch.bfloat16), dtype=BF16, layout=RM),
                ttnn.from_torch(tm.bias.detach().reshape(1, 1, 1, out_ch).to(torch.bfloat16), dtype=BF16, layout=RM),
            )
        wt, bt = self._cw[key]
        return ttnn.conv2d(
            input_tensor=x,
            weight_tensor=wt,
            bias_tensor=bt,
            device=self.device,
            in_channels=in_ch,
            out_channels=out_ch,
            batch_size=1,
            input_height=h,
            input_width=w,
            kernel_size=(k, k),
            stride=(s, s),
            padding=pad,
            dilation=(1, 1),
            groups=1,
            conv_config=ttnn.Conv2dConfig(weights_dtype=BF16),
            compute_config=self._ck(),
            dtype=F32,
            memory_config=DRAM,
        )

    def __call__(self, hidden_states, output_size=None, **_ignored):
        sample = hidden_states
        if isinstance(sample, ttnn.Tensor):
            B, C, H, W = list(sample.shape)
            if sample.dtype != F32:
                if sample.layout != TILE:
                    sample = ttnn.to_layout(sample, TILE)
                sample = ttnn.typecast(sample, F32)
            if sample.layout != RM:
                sample = ttnn.to_layout(sample, RM)
            x = ttnn.permute(sample, [0, 2, 3, 1])  # NHWC [1,H,W,C]
        else:
            t = sample.to(torch.float32)
            B, C, H, W = t.shape
            x = ttnn.from_torch(
                t.permute(0, 2, 3, 1).contiguous(), dtype=F32, layout=RM, device=self.device, memory_config=DRAM
            )

        if x.layout != RM:
            x = ttnn.to_layout(x, RM)
        x = ttnn.upsample(x, scale_factor=2)  # nearest 2x -> [1,2H,2W,C]
        Ho, Wo = H * 2, W * 2
        x = ttnn.reshape(x, [1, 1, Ho * Wo, C])

        if self.conv is not None:
            oc = int(self.conv.out_channels)
            k = int(self.conv.kernel_size[0])
            p = int(self.conv.padding[0]) if hasattr(self.conv.padding, "__len__") else int(self.conv.padding)
            x = self._conv(x, self.conv, C, oc, Ho, Wo, k=k, s=1, pad=(p, p))
        else:
            oc = C

        nhwc = ttnn.reshape(x, [1, Ho, Wo, oc])
        if nhwc.layout != RM:
            nhwc = ttnn.to_layout(nhwc, RM)
        return ttnn.permute(nhwc, [0, 3, 1, 2])  # [1, oc, Ho, Wo]


def build(device, torch_module):
    """PCC-harness entry point: native TTNN diffusers Upsample2D."""
    return _Upsample2D(device, torch_module)

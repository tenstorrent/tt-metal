# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `downsample2_d` (diffusers ``Downsample2D``) for
meituan-longcat/LongCat-Image (VAE encoder downsample).

``Downsample2D.forward(hidden_states)`` (use_conv=True, padding=0)::

    hidden_states = F.pad(hidden_states, (0,1,0,1))   # asymmetric: pad right+bottom
    hidden_states = conv(hidden_states)               # 3x3 stride-2, padding=0
    return hidden_states

fp32 activations + fp32 conv output, bf16 conv weights, HiFi4/fp32-accumulate —
identical precision policy to the graduated autoencoder_k_l / decoder ports.
Tensor format between ops is NHWC flattened to [1,1,H*W,C]; I/O is NCHW.
"""

from __future__ import annotations

import torch

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
F32 = ttnn.float32
BF16 = ttnn.bfloat16
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT


class _Downsample2D:
    def __init__(self, device, ds):
        self.device = device
        self.ds = ds.eval() if hasattr(ds, "eval") else ds
        self.channels = int(self.ds.conv.in_channels)
        self.stride = (
            int(self.ds.conv.stride[0]) if hasattr(self.ds.conv.stride, "__len__") else int(self.ds.conv.stride)
        )
        # diffusers uses asymmetric (0,1,0,1) F.pad only when the module's own pad is 0.
        self.asym = int(getattr(self.ds, "padding", 0)) == 0
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

    def __call__(self, hidden_states, **_ignored):
        sample = hidden_states
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

        k = int(self.ds.conv.kernel_size[0])
        if self.asym:
            pad = (0, 1, 0, 1)  # pad bottom + right (top,bottom,left,right)
        else:
            p = int(getattr(self.ds, "padding", 1))
            pad = (p, p)
        x = self._conv(x, self.ds.conv, self.channels, self.channels, H, W, k=k, s=self.stride, pad=pad)

        Ho, Wo = H // self.stride, W // self.stride
        nhwc = ttnn.reshape(x, [1, Ho, Wo, self.channels])
        if nhwc.layout != RM:
            nhwc = ttnn.to_layout(nhwc, RM)
        nchw = ttnn.permute(nhwc, [0, 3, 1, 2])  # [1, C, Ho, Wo]
        return nchw


def build(device, torch_module):
    """PCC-harness entry point: native TTNN Downsample2D from the torch module."""
    return _Downsample2D(device, torch_module)

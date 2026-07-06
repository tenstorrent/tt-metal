# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `down_encoder_block2_d` (diffusers ``DownEncoderBlock2D``)
for meituan-longcat/LongCat-Image (VAE encoder down block).

``DownEncoderBlock2D.forward(hidden_states)``::

    for resnet in resnets:  hidden_states = resnet(hidden_states)
    if downsamplers is not None:
        for ds in downsamplers: hidden_states = ds(hidden_states)   # 3x3 stride-2 conv
    return hidden_states

Precision matches the graduated autoencoder_k_l / decoder ports: fp32 everywhere
except conv WEIGHTS (bf16 — Wormhole truncates fp32 conv weights to bf16 and
processes bf16 at full mantissa via HiFi4). GroupNorm is a fully-fp32 manual
implementation (ttnn.group_norm is bf16-only and is the VAE precision limiter).
Downsample uses diffusers' asymmetric (0,1,0,1) pad. Tensor format between ops is
NHWC flattened to [1,1,H*W,C]; the block I/O is NCHW to match the torch reference.
"""

from __future__ import annotations

import torch

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
F32 = ttnn.float32
BF16 = ttnn.bfloat16
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT


class _DownEncoderBlock2D:
    def __init__(self, device, block):
        self.device = device
        self.block = block.eval() if hasattr(block, "eval") else block
        self._cw = {}
        self._gn = {}
        self._compute = None

    def _ck(self):
        if self._compute is None:
            self._compute = ttnn.init_device_compute_kernel_config(
                self.device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
            )
        return self._compute

    def _conv(self, x, tm, in_ch, out_ch, h, w, k=3, s=1, pad=(1, 1)):
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

    def _gn_params(self, gn, C):
        # Fully-fp32 GroupNorm params; A[c,c']=1/cpg block-averaging matrix (exact in bf16).
        key = id(gn)
        if key not in self._gn:
            cpg = C // 32
            A = torch.zeros(C, C, dtype=torch.float32)
            for g in range(32):
                A[g * cpg : (g + 1) * cpg, g * cpg : (g + 1) * cpg] = 1.0 / cpg
            self._gn[key] = (
                ttnn.from_torch(A, dtype=BF16, layout=TILE, device=self.device, memory_config=DRAM),
                ttnn.from_torch(
                    gn.weight.detach().reshape(1, 1, 1, C).to(torch.float32),
                    dtype=F32,
                    layout=TILE,
                    device=self.device,
                    memory_config=DRAM,
                ),
                ttnn.from_torch(
                    gn.bias.detach().reshape(1, 1, 1, C).to(torch.float32),
                    dtype=F32,
                    layout=TILE,
                    device=self.device,
                    memory_config=DRAM,
                ),
                float(gn.eps),
            )
        return self._gn[key]

    def _gn_silu(self, x, gn, C, H, W, silu=True):
        A, wt, bt, eps = self._gn_params(gn, C)
        HW = H * W
        x = ttnn.reshape(x, [1, 1, HW, C])
        if x.layout != TILE:
            x = ttnn.to_layout(x, TILE)
        if x.dtype != F32:
            x = ttnn.typecast(x, F32)
        mu_c = ttnn.mean(x, dim=2, keepdim=True, compute_kernel_config=self._ck())
        msq_c = ttnn.mean(ttnn.mul(x, x), dim=2, keepdim=True, compute_kernel_config=self._ck())
        mu_g = ttnn.matmul(mu_c, A, compute_kernel_config=self._ck(), dtype=F32)
        msq_g = ttnn.matmul(msq_c, A, compute_kernel_config=self._ck(), dtype=F32)
        var_g = ttnn.subtract(msq_g, ttnn.mul(mu_g, mu_g))
        inv = ttnn.rsqrt(ttnn.add(var_g, eps))
        scale = ttnn.mul(wt, inv)
        shift = ttnn.subtract(bt, ttnn.mul(mu_g, scale))
        out = ttnn.add(ttnn.mul(x, scale), shift)
        if silu:
            out = ttnn.silu(out)
        return out

    def _resnet(self, x, rb, in_ch, out_ch, H, W):
        h1 = self._gn_silu(x, rb.norm1, in_ch, H, W)
        c1 = self._conv(h1, rb.conv1, in_ch, out_ch, H, W)
        h2 = self._gn_silu(c1, rb.norm2, out_ch, H, W)
        c2 = self._conv(h2, rb.conv2, out_ch, out_ch, H, W)
        sc = (
            self._conv(x, rb.conv_shortcut, in_ch, out_ch, H, W, k=1, pad=(0, 0))
            if getattr(rb, "conv_shortcut", None) is not None
            else x
        )
        if c2.layout != sc.layout:
            sc = ttnn.to_layout(sc, c2.layout)
        return ttnn.add(c2, sc)

    def _downsample(self, x, ds, C, H, W):
        # diffusers Downsample2D: F.pad(x,(0,1,0,1)) then 3x3 stride-2 conv, padding=0.
        return self._conv(x, ds.conv, C, C, H, W, k=3, s=2, pad=(0, 1, 0, 1))

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

        cc = C
        for rb in self.block.resnets:
            oc = int(rb.conv2.out_channels)
            x = self._resnet(x, rb, cc, oc, H, W)
            cc = oc
        downs = getattr(self.block, "downsamplers", None)
        if downs is not None and len(downs) > 0:
            for ds in downs:
                x = self._downsample(x, ds, cc, H, W)
                H, W = H // 2, W // 2

        nhwc = ttnn.reshape(x, [1, H, W, cc])
        if nhwc.layout != RM:
            nhwc = ttnn.to_layout(nhwc, RM)
        nchw = ttnn.permute(nhwc, [0, 3, 1, 2])  # [1, cc, H, W]
        return nchw


def build(device, torch_module):
    """PCC-harness entry point: native TTNN DownEncoderBlock2D from the torch module."""
    return _DownEncoderBlock2D(device, torch_module)

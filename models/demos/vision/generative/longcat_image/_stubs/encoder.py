# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `encoder` (diffusers VAE ``Encoder``) for
meituan-longcat/LongCat-Image.

``Encoder.forward(sample)``::

    sample = conv_in(sample)                    # 3 -> 128
    for down in down_blocks: sample = down(sample)   # 128,256,512,512 (first 3 downsample)
    sample = mid_block(sample)                  # resnet, attention, resnet @ 512
    sample = conv_norm_out(sample); silu        # GroupNorm(32) @ 512
    sample = conv_out(sample)                   # 512 -> 2*latent_channels (32)
    return sample

Precision identical to the graduated autoencoder_k_l / decoder ports: fp32
everywhere except conv WEIGHTS (bf16 — Wormhole truncates fp32 conv weights to
bf16 and processes bf16 at full mantissa via HiFi4). GroupNorm is a fully-fp32
manual implementation (ttnn.group_norm is bf16-only and is the VAE precision
limiter). Downsample uses diffusers' asymmetric (0,1,0,1) pad. Tensor format
between ops is NHWC flattened to [1,1,H*W,C]; the encoder I/O is NCHW.
"""

from __future__ import annotations

import torch

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
F32 = ttnn.float32
BF16 = ttnn.bfloat16
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT


class _Encoder:
    def __init__(self, device, encoder):
        self.device = device
        self.enc = encoder.eval() if hasattr(encoder, "eval") else encoder
        self._cw = {}
        self._gn = {}
        self._lin = {}
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
        return self._conv(x, ds.conv, C, C, H, W, k=3, s=2, pad=(0, 1, 0, 1))

    def _linear(self, x, tm):
        key = id(tm)
        if key not in self._lin:
            self._lin[key] = (
                ttnn.from_torch(
                    tm.weight.detach().to(torch.float32), dtype=F32, layout=TILE, device=self.device, memory_config=DRAM
                ),
                ttnn.from_torch(
                    tm.bias.detach().reshape(1, 1, -1).to(torch.float32),
                    dtype=F32,
                    layout=TILE,
                    device=self.device,
                    memory_config=DRAM,
                ),
            )
        wt, bt = self._lin[key]
        return ttnn.linear(
            x, wt, bias=bt, transpose_b=True, memory_config=DRAM, dtype=F32, compute_kernel_config=self._ck()
        )

    def _attention(self, x, attn, C, H, W):
        hw = H * W
        normed = self._gn_silu(x, attn.group_norm, C, H, W, silu=False)
        h = ttnn.reshape(normed, [1, hw, C])
        q = self._linear(h, attn.to_q)
        k = self._linear(h, attn.to_k)
        v = self._linear(h, attn.to_v)
        scores = ttnn.matmul(q, k, transpose_b=True, dtype=F32, compute_kernel_config=self._ck(), memory_config=DRAM)
        scores = ttnn.multiply(scores, C**-0.5)
        probs = ttnn.softmax(scores, dim=-1)
        a = ttnn.matmul(probs, v, dtype=F32, compute_kernel_config=self._ck(), memory_config=DRAM)
        o = ttnn.reshape(self._linear(a, attn.to_out[0]), [1, 1, hw, C])
        if x.layout != o.layout:
            x = ttnn.to_layout(x, o.layout)
        return ttnn.add(o, x)

    def _midblock(self, x, mb, C, H, W):
        x = self._resnet(x, mb.resnets[0], C, C, H, W)
        x = self._attention(x, mb.attentions[0], C, H, W)
        x = self._resnet(x, mb.resnets[1], C, C, H, W)
        return x

    def _down_block(self, x, db, in_ch, H, W):
        cc = in_ch
        for rb in db.resnets:
            oc = int(rb.conv2.out_channels)
            x = self._resnet(x, rb, cc, oc, H, W)
            cc = oc
        downs = getattr(db, "downsamplers", None)
        if downs is not None and len(downs) > 0:
            for ds in downs:
                x = self._downsample(x, ds, cc, H, W)
                H, W = H // 2, W // 2
        return x, cc, H, W

    def __call__(self, sample, **_ignored):
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

        e = self.enc
        cc = int(e.conv_in.out_channels)
        x = self._conv(x, e.conv_in, C, cc, H, W)
        for db in e.down_blocks:
            x, cc, H, W = self._down_block(x, db, cc, H, W)
        x = self._midblock(x, e.mid_block, cc, H, W)
        x = self._gn_silu(x, e.conv_norm_out, cc, H, W, silu=True)
        oc = int(e.conv_out.out_channels)
        x = self._conv(x, e.conv_out, cc, oc, H, W)

        nhwc = ttnn.reshape(x, [1, H, W, oc])
        if nhwc.layout != RM:
            nhwc = ttnn.to_layout(nhwc, RM)
        nchw = ttnn.permute(nhwc, [0, 3, 1, 2])  # [1, oc, H, W]
        return nchw


def build(device, torch_module):
    """PCC-harness entry point: native TTNN VAE Encoder from the torch module."""
    return _Encoder(device, torch_module)

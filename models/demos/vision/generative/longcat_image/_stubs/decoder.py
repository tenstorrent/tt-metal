# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `decoder` (diffusers VAE ``Decoder``) for
meituan-longcat/LongCat-Image.

``Decoder.forward(sample, latent_embeds=None)`` (latent_embeds is None for this
VAE — no spatial norm)::

    sample = conv_in(sample)                 # latent_channels -> 512
    sample = mid_block(sample)               # resnet, attention, resnet @ 512
    for up in up_blocks: sample = up(sample) # 512,512,256,128 (first 3 upsample x2)
    sample = conv_norm_out(sample); silu     # GroupNorm(32) @ 128
    sample = conv_out(sample)                # 128 -> 3
    return DecoderOutput(sample=sample)

Config mirrors the VAE: block_out_channels=[128,256,512,512] (decoder walks them
reversed -> [512,512,256,128]), layers_per_block=2 (=> 3 resnets per up block),
norm_num_groups=32, GroupNorm eps=1e-6, act=silu.

Precision (identical to the graduated autoencoder_k_l port): everything runs fp32
except conv WEIGHTS (bf16 — Wormhole processes those at full mantissa via HiFi4 and
truncates native fp32 conv weights to bf16 anyway). GroupNorm is a fully-fp32 manual
implementation because ttnn.group_norm hard-requires bf16 input+affine and is the
documented VAE precision limiter. Tensor format between ops is NHWC flattened to
[1,1,H*W,C]; the input latent and returned image are NCHW to match the torch ref.
"""

from __future__ import annotations

import torch

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
F32 = ttnn.float32
BF16 = ttnn.bfloat16
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT


class _Decoder:
    def __init__(self, device, decoder):
        self.device = device
        self.dec = decoder.eval() if hasattr(decoder, "eval") else decoder
        self.latent_channels = int(self.dec.conv_in.in_channels)  # 16
        self._cw = {}  # conv weight/bias cache (keyed by id(module))
        self._gn = {}  # group-norm A/weight/bias/eps cache
        self._lin = {}  # linear weight/bias cache
        self._compute = None

    # ── low-level helpers (identical to the graduated autoencoder_k_l port) ──
    def _ck(self):
        if self._compute is None:
            self._compute = ttnn.init_device_compute_kernel_config(
                self.device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
            )
        return self._compute

    def _conv(self, x, tm, in_ch, out_ch, h, w, k=3, s=1, pad=(1, 1)):
        # bf16 weights + fp32 conv OUTPUT (residual stream never truncates
        # per-layer); HiFi4 + fp32-accumulate compute config.
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
        # Fully-fp32 GroupNorm params. `A` [C,C] block-averaging matrix:
        # A[c,c']=1/cpg if same group else 0 (cpg=C/32 in {4,8,16} -> 0.25/0.125/
        # 0.0625, all exact in bf16 -> lossless matmul).
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
        # Native fp32 GroupNorm (num_groups=32) — avoids bf16-only ttnn.group_norm.
        A, wt, bt, eps = self._gn_params(gn, C)
        HW = H * W
        x = ttnn.reshape(x, [1, 1, HW, C])
        if x.layout != TILE:
            x = ttnn.to_layout(x, TILE)
        if x.dtype != F32:
            x = ttnn.typecast(x, F32)
        mu_c = ttnn.mean(x, dim=2, keepdim=True, compute_kernel_config=self._ck())  # [1,1,1,C]
        msq_c = ttnn.mean(ttnn.mul(x, x), dim=2, keepdim=True, compute_kernel_config=self._ck())  # [1,1,1,C]
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

    def _upsample(self, x, up, C, H, W):
        x = ttnn.reshape(x, [1, H, W, C])
        if x.layout != RM:
            x = ttnn.to_layout(x, RM)
        u = ttnn.upsample(x, scale_factor=2)  # nearest, [1,2H,2W,C]
        u = ttnn.reshape(u, [1, 1, 4 * H * W, C])
        return self._conv(u, up.conv, C, C, 2 * H, 2 * W, k=3, s=1, pad=(1, 1))

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
        # Manual fp32 attention (1 head, scale=C**-0.5, residual, no rescale).
        hw = H * W
        normed = self._gn_silu(x, attn.group_norm, C, H, W, silu=False)  # [1,1,hw,C]
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

    def _up_block(self, x, ub, in_ch, out_ch, H, W, has_up):
        x = self._resnet(x, ub.resnets[0], in_ch, out_ch, H, W)
        x = self._resnet(x, ub.resnets[1], out_ch, out_ch, H, W)
        x = self._resnet(x, ub.resnets[2], out_ch, out_ch, H, W)
        if has_up:
            x = self._upsample(x, ub.upsamplers[0], out_ch, H, W)
        return x

    def __call__(self, sample, latent_embeds=None, return_dict=True, **_ignored):
        # sample: ttnn.Tensor or torch.Tensor, NCHW (B, latent_channels, H, W)
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

        d = self.dec
        x = self._conv(x, d.conv_in, self.latent_channels, 512, H, W)
        x = self._midblock(x, d.mid_block, 512, H, W)
        x = self._up_block(x, d.up_blocks[0], 512, 512, H, W, True)
        x = self._up_block(x, d.up_blocks[1], 512, 512, H * 2, W * 2, True)
        x = self._up_block(x, d.up_blocks[2], 512, 256, H * 4, W * 4, True)
        x = self._up_block(x, d.up_blocks[3], 256, 128, H * 8, W * 8, False)
        Hf, Wf = H * 8, W * 8
        x = self._gn_silu(x, d.conv_norm_out, 128, Hf, Wf, silu=True)
        x = self._conv(x, d.conv_out, 128, 3, Hf, Wf)

        nhwc = ttnn.reshape(x, [1, Hf, Wf, 3])
        if nhwc.layout != RM:
            nhwc = ttnn.to_layout(nhwc, RM)
        nchw = ttnn.permute(nhwc, [0, 3, 1, 2])  # [1,3,Hf,Wf]
        return nchw


def build(device, torch_module):
    """PCC-harness entry point: native TTNN VAE Decoder from the torch module."""
    return _Decoder(device, torch_module)

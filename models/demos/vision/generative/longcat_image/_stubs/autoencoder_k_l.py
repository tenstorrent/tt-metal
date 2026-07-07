# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `autoencoder_k_l` (diffusers ``AutoencoderKL``, the Flux-style VAE)
for meituan-longcat/LongCat-Image.

``AutoencoderKL.forward(sample, sample_posterior=False)``::

    h        = encoder(sample)             # no quant_conv (use_quant_conv=False)
    mean, _  = chunk(h, 2, dim=1)          # posterior.mode() == mean (first latent_channels)
    dec      = decoder(mean)               # no post_quant_conv (use_post_quant_conv=False)
    return DecoderOutput(sample=dec)

Config: in=3, out=3, latent_channels=16, block_out_channels=[128,256,512,512],
layers_per_block=2, norm_num_groups=32, GroupNorm eps=1e-6, act=silu.

Everything runs natively in TTNN:
  * conv2d  -> ttnn.conv2d (NHWC "conv format" [1,1,H*W,C]; bf16 weights, fp32-accumulate).
  * groupnorm -> ttnn.group_norm (bf16 in, fp32 internal via input_mask; adaptive DRAM core-grid).
  * silu    -> ttnn.silu.
  * attention (UNetMidBlock2D) -> group_norm + to_q/k/v linears + SDPA (1 head, scale=C**-0.5) + to_out.
  * downsample -> strided (2,2) 3x3 conv with diffusers' asymmetric (0,1,0,1) pad.
  * upsample   -> ttnn.upsample(nearest, x2) + 3x3 conv.

Tensor format: NHWC flattened to [1, 1, H*W, C] between ops; the input image and the returned
reconstruction are NCHW [1, C, H, W] to match the torch reference.
"""

from __future__ import annotations

import torch

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
F32 = ttnn.float32
BF16 = ttnn.bfloat16
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT


class _AutoencoderKL:
    def __init__(self, device, vae):
        self.device = device
        self.vae = vae.eval() if hasattr(vae, "eval") else vae
        self.latent_channels = int(getattr(vae.config, "latent_channels", 16))
        self._cw = {}  # conv weight/bias cache (keyed by id(module))
        self._gn = {}  # group-norm weight/bias/mask cache
        self._cg = {}  # group-norm core-grid cache (keyed by (C,H,W))
        self._lin = {}  # linear weight/bias cache
        self._compute = None

    # ── low-level helpers ────────────────────────────────────────────────
    def _ck(self):
        if self._compute is None:
            self._compute = ttnn.init_device_compute_kernel_config(
                self.device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
            )
        return self._compute

    def _conv(self, x, tm, in_ch, out_ch, h, w, k=3, s=1, pad=(1, 1)):
        # bf16 weights (Wormhole processes these at full mantissa via HiFi4's
        # 4-pass decomposition — more accurate than native fp32 weights on WH),
        # but fp32 conv OUTPUT so the residual stream never suffers per-layer
        # bf16 truncation. fp32-accumulate compute config throughout.
        # Pre-PREPARE (tilize/reformat) the conv weights ONCE and cache them, so
        # the traced forward calls conv2d with already-prepared weights and does
        # NO host->device write inside begin/end_trace_capture. Passing a raw
        # ROW_MAJOR weight makes conv2d re-prepare (and write) on every call,
        # which trips "TT_FATAL: Writes are not supported during trace capture"
        # in the VAE-decode stage. Keyed by (module, input geometry) because the
        # prepared layout depends on input_height/width/memory_config/dtype.
        cfg = ttnn.Conv2dConfig(weights_dtype=BF16)
        ck = self._ck()
        key = (id(tm), int(h), int(w))
        if key not in self._cw:
            w_raw = ttnn.from_torch(tm.weight.detach().to(torch.bfloat16), dtype=BF16, layout=RM)
            b_raw = ttnn.from_torch(tm.bias.detach().reshape(1, 1, 1, out_ch).to(torch.bfloat16), dtype=BF16, layout=RM)
            pw = ttnn.prepare_conv_weights(
                weight_tensor=w_raw, input_memory_config=x.memory_config(), input_layout=x.layout,
                weights_format="OIHW", in_channels=in_ch, out_channels=out_ch, batch_size=1,
                input_height=h, input_width=w, kernel_size=(k, k), stride=(s, s), padding=pad,
                dilation=(1, 1), has_bias=True, groups=1, input_dtype=x.dtype, device=self.device,
                conv_config=cfg, compute_config=ck,
            )
            pb = ttnn.prepare_conv_bias(
                bias_tensor=b_raw, input_memory_config=x.memory_config(), input_layout=x.layout,
                in_channels=in_ch, out_channels=out_ch, batch_size=1, input_height=h, input_width=w,
                kernel_size=(k, k), stride=(s, s), padding=pad, dilation=(1, 1), groups=1,
                input_dtype=x.dtype, device=self.device, conv_config=cfg, compute_config=ck,
            )
            self._cw[key] = (pw, pb)
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
            conv_config=cfg,
            compute_config=ck,
            dtype=F32,
            memory_config=DRAM,
        )

    def _gn_params(self, gn, C):
        # Fully-fp32 GroupNorm params. `A` is a [C,C] block-averaging matrix:
        # A[c,c'] = 1/cpg if c,c' are in the same group else 0, so `mu_c @ A`
        # broadcasts each group's mean back onto its channels in one matmul.
        # The 1/cpg entries (cpg=C/32 in {4,8,16} -> 0.25/0.125/0.0625) are all
        # exact powers of two, hence exact in bf16 -> the matmul loses nothing.
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
        # Native fp32 GroupNorm (num_groups=32). Avoids ttnn.group_norm, which
        # hard-requires bf16 input + bf16 affine and is the documented precision
        # limiter for VAEs. All statistics/affine here stay fp32.
        A, wt, bt, eps = self._gn_params(gn, C)
        HW = H * W
        x = ttnn.reshape(x, [1, 1, HW, C])
        if x.layout != TILE:
            x = ttnn.to_layout(x, TILE)
        if x.dtype != F32:
            x = ttnn.typecast(x, F32)
        # per-channel mean and mean-of-squares over the HW axis (dim=2)
        mu_c = ttnn.mean(x, dim=2, keepdim=True, compute_kernel_config=self._ck())  # [1,1,1,C]
        msq_c = ttnn.mean(ttnn.mul(x, x), dim=2, keepdim=True, compute_kernel_config=self._ck())  # [1,1,1,C]
        # average the cpg channels of each group, broadcast back per-channel
        mu_g = ttnn.matmul(mu_c, A, compute_kernel_config=self._ck(), dtype=F32)  # [1,1,1,C]
        msq_g = ttnn.matmul(msq_c, A, compute_kernel_config=self._ck(), dtype=F32)  # [1,1,1,C]
        var_g = ttnn.subtract(msq_g, ttnn.mul(mu_g, mu_g))  # E[x^2]-E[x]^2 (biased)
        inv = ttnn.rsqrt(ttnn.add(var_g, eps))  # 1/sqrt(var+eps)
        scale = ttnn.mul(wt, inv)  # weight/sqrt(var+eps)
        shift = ttnn.subtract(bt, ttnn.mul(mu_g, scale))  # bias - mean*scale
        out = ttnn.add(ttnn.mul(x, scale), shift)  # (x-mean)/std*w+b, bcast HW
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
        # Manual fp32 attention (1 head): more precise than bf16 SDPA on this VAE.
        hw = H * W
        normed = self._gn_silu(x, attn.group_norm, C, H, W, silu=False)  # [1,1,hw,C]
        h = ttnn.reshape(normed, [1, hw, C])
        q = self._linear(h, attn.to_q)  # [1,hw,C] fp32
        k = self._linear(h, attn.to_k)
        v = self._linear(h, attn.to_v)
        scores = ttnn.matmul(q, k, transpose_b=True, dtype=F32, compute_kernel_config=self._ck(), memory_config=DRAM)
        scores = ttnn.multiply(scores, C**-0.5)
        probs = ttnn.softmax(scores, dim=-1)
        a = ttnn.matmul(probs, v, dtype=F32, compute_kernel_config=self._ck(), memory_config=DRAM)  # [1,hw,C]
        o = ttnn.reshape(self._linear(a, attn.to_out[0]), [1, 1, hw, C])
        # everything stays fp32 through the residual add
        if x.layout != o.layout:
            x = ttnn.to_layout(x, o.layout)
        return ttnn.add(o, x)

    def _midblock(self, x, mb, C, H, W):
        x = self._resnet(x, mb.resnets[0], C, C, H, W)
        x = self._attention(x, mb.attentions[0], C, H, W)
        x = self._resnet(x, mb.resnets[1], C, C, H, W)
        return x

    def _down_block(self, x, db, in_ch, out_ch, H, W, has_down):
        x = self._resnet(x, db.resnets[0], in_ch, out_ch, H, W)
        x = self._resnet(x, db.resnets[1], out_ch, out_ch, H, W)
        if has_down:
            x = self._downsample(x, db.downsamplers[0], out_ch, H, W)
        return x

    def _up_block(self, x, ub, in_ch, out_ch, H, W, has_up):
        x = self._resnet(x, ub.resnets[0], in_ch, out_ch, H, W)
        x = self._resnet(x, ub.resnets[1], out_ch, out_ch, H, W)
        x = self._resnet(x, ub.resnets[2], out_ch, out_ch, H, W)
        if has_up:
            x = self._upsample(x, ub.upsamplers[0], out_ch, H, W)
        return x

    # ── encode / decode ──────────────────────────────────────────────────
    def _encode(self, x_cf, H, W):
        e = self.vae.encoder
        # thread height AND width independently through the 3 downsamples (non-square inputs:
        # the edit path derives H!=W from the image aspect ratio via calculate_dimensions).
        h2, h4, h8 = H // 2, H // 4, H // 8
        w2, w4, w8 = W // 2, W // 4, W // 8
        x = self._conv(x_cf, e.conv_in, 3, 128, H, W)
        x = self._down_block(x, e.down_blocks[0], 128, 128, H, W, True)
        x = self._down_block(x, e.down_blocks[1], 128, 256, h2, w2, True)
        x = self._down_block(x, e.down_blocks[2], 256, 512, h4, w4, True)
        x = self._down_block(x, e.down_blocks[3], 512, 512, h8, w8, False)
        x = self._midblock(x, e.mid_block, 512, h8, w8)
        x = self._gn_silu(x, e.conv_norm_out, 512, h8, w8, silu=True)
        x = self._conv(x, e.conv_out, 512, 2 * self.latent_channels, h8, w8)
        return x, h8, w8

    def _decode(self, z, hb, wb):
        d = self.vae.decoder
        x = self._conv(z, d.conv_in, self.latent_channels, 512, hb, wb)
        x = self._midblock(x, d.mid_block, 512, hb, wb)
        x = self._up_block(x, d.up_blocks[0], 512, 512, hb, wb, True)
        x = self._up_block(x, d.up_blocks[1], 512, 512, hb * 2, wb * 2, True)
        x = self._up_block(x, d.up_blocks[2], 512, 256, hb * 4, wb * 4, True)
        x = self._up_block(x, d.up_blocks[3], 256, 128, hb * 8, wb * 8, False)
        Hf, Wf = hb * 8, wb * 8
        x = self._gn_silu(x, d.conv_norm_out, 128, Hf, Wf, silu=True)
        x = self._conv(x, d.conv_out, 128, 3, Hf, Wf)
        return x, Hf, Wf

    def __call__(self, sample, sample_posterior=False, return_dict=True, generator=None, **_ignored):
        # sample: ttnn.Tensor or torch.Tensor, NCHW (B, 3, H, W)
        if isinstance(sample, ttnn.Tensor):
            shp = list(sample.shape)
            B, C, H, W = shp
            # harness hands us bf16; lift to fp32 up front so the entire VAE runs fp32
            if sample.dtype != F32:
                if sample.layout != TILE:
                    sample = ttnn.to_layout(sample, TILE)
                sample = ttnn.typecast(sample, F32)
            if sample.layout != RM:
                sample = ttnn.to_layout(sample, RM)
            nhwc = ttnn.permute(sample, [0, 2, 3, 1])
            x_cf = ttnn.reshape(nhwc, [1, 1, H * W, C])
        else:
            t = sample.to(torch.float32)
            B, C, H, W = t.shape
            x_cf = ttnn.from_torch(
                t.permute(0, 2, 3, 1).reshape(1, 1, H * W, C).contiguous(),
                dtype=F32,
                layout=RM,
                device=self.device,
                memory_config=DRAM,
            )

        moments, hb = self._encode(x_cf, H, W)  # [1,1,hb*wb, 2*lat]
        wb = hb
        mom = ttnn.reshape(moments, [1, 1, hb * wb, 2 * self.latent_channels])
        if mom.layout != TILE:
            mom = ttnn.to_layout(mom, TILE)
        # posterior.mode() == mean == first `latent_channels` channels
        z = ttnn.slice(mom, [0, 0, 0, 0], [1, 1, hb * wb, self.latent_channels], [1, 1, 1, 1])

        dec, Hf, Wf = self._decode(z, hb, wb)  # [1,1,Hf*Wf,3]
        nhwc = ttnn.reshape(dec, [1, Hf, Wf, 3])
        if nhwc.layout != RM:
            nhwc = ttnn.to_layout(nhwc, RM)
        nchw = ttnn.permute(nhwc, [0, 3, 1, 2])  # [1,3,Hf,Wf]
        return nchw


def build(device, torch_module):
    """PCC-harness entry point: native TTNN AutoencoderKL from the torch VAE module."""
    return _AutoencoderKL(device, torch_module)

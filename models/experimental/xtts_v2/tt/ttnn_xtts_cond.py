# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of the XTTS-v2 conditioning branch (Block 1).

Reference: models/experimental/xtts_v2/reference/xtts_cond_ref.py

This file currently implements the **Perceiver resampler** (the tail of the conditioning
branch). The conditioning encoder (Conv1d + 6x GroupNorm attention over T) is TODO — its
GroupNorm pools over the time axis (T=505, not tile-aligned), the known hard part.

PerceiverResampler (dim=1024, depth=2, 32 latents, 8 heads x 64):
  latents [1,32,1024]; per layer:
    cross-attn: context = concat([latents, frames]); q=to_q(latents), k,v=to_kv(context);
                8 heads x 64, scale 1/8, softmax over keys; to_out; + residual
    FFN: Linear(1024->5460) -> GEGLU -> Linear(2730->1024); + residual
  RMSNorm: normalize(x,dim=-1) * sqrt(1024) * gamma

Padding: frames (T=505) are padded to a tile multiple (512) so the concat/attention stay
tile-aligned; padded key positions are masked to -inf in the attention softmax.
"""

import ttnn

import torch

from models.experimental.xtts_v2.reference.xtts_cond_ref import load_cond_state

DIM = 1024
HEADS = 8
DH = 64
INNER = HEADS * DH  # 512
LATENTS = 32
DEPTH = 2

# conditioning encoder
ENC_HEADS = 16
ENC_DH = 64
ENC_BLOCKS = 6
GN_GROUPS = 32
GN_CH = DIM // GN_GROUPS  # 32 channels per group


def _compute_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def preprocess_perceiver_parameters(device, ckpt_path=None, dtype=ttnn.float32):
    w = load_cond_state(ckpt_path) if ckpt_path else load_cond_state()

    def linT(name):  # nn.Linear weight [out,in] -> ttnn [in,out] for x@W
        return ttnn.from_torch(w[name].t().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    def t(x):
        return ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    params = {
        "latents": t(w["latents"].unsqueeze(0)),  # [1,32,1024]
        "norm_gamma": t(w["norm.gamma"].reshape(1, -1)),  # [1,1024]
        "layers": [],
    }
    for i in range(DEPTH):
        params["layers"].append(
            {
                "to_q": linT(f"layers.{i}.0.to_q.weight"),
                "to_kv": linT(f"layers.{i}.0.to_kv.weight"),
                "to_out": linT(f"layers.{i}.0.to_out.weight"),
                "ff1_w": linT(f"layers.{i}.1.0.weight"),
                "ff1_b": t(w[f"layers.{i}.1.0.bias"].reshape(1, -1)),
                "ff2_w": linT(f"layers.{i}.1.2.weight"),
                "ff2_b": t(w[f"layers.{i}.1.2.bias"].reshape(1, -1)),
            }
        )
    return params


def preprocess_encoder_parameters(device, ckpt_path=None, dtype=ttnn.float32):
    w = load_cond_state(ckpt_path) if ckpt_path else load_cond_state()

    def convT(name):  # Conv1d weight [out,in,1] -> ttnn [in,out] for x@W
        return ttnn.from_torch(
            w[name].squeeze(-1).t().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

    def vec(x):
        return ttnn.from_torch(x.reshape(1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    params = {
        "init_w": convT("init.weight"),
        "init_b": vec(w["init.bias"]),
        "blocks": [],
    }
    for i in range(ENC_BLOCKS):
        params["blocks"].append(
            {
                "gn_w": vec(w[f"attn.{i}.norm.weight"]),
                "gn_b": vec(w[f"attn.{i}.norm.bias"]),
                "qkv_w": convT(f"attn.{i}.qkv.weight"),
                "qkv_b": vec(w[f"attn.{i}.qkv.bias"]),
                "proj_w": convT(f"attn.{i}.proj_out.weight"),
                "proj_b": vec(w[f"attn.{i}.proj_out.bias"]),
            }
        )
    return params


class TTNNConditioningEncoder:
    """mel [1, S, 80] (S padded to tile mult, T_real real frames) -> enc [1, S, 1024].
    Padded time rows carry garbage (bias) and are masked in group-norm stats + attention."""

    def __init__(self, device, parameters, t_real, s_pad):
        self.device = device
        self.p = parameters
        self.cc = _compute_config()
        self.t_real = t_real
        self.s_pad = s_pad
        self.scale = ENC_DH**-0.5
        # time mask [1, S, 1]: 1 for real frames, 0 for padded
        tm = torch.zeros(1, s_pad, 1)
        tm[:, :t_real, :] = 1.0
        self.time_mask = ttnn.from_torch(tm, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        # additive key mask [1,1,1,S]: 0 for real keys, -inf for padded
        km = torch.zeros(1, 1, 1, s_pad)
        km[:, :, :, t_real:] = -1e9
        self.key_mask = ttnn.from_torch(km, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        self.n = float(t_real * GN_CH)  # elements per group (real T x channels/group)

    def _group_norm(self, x, gn_w, gn_b):
        # x [1, S, 1024]; GroupNorm(32,1024) pooled over (real T x 32 ch/group)
        xm = ttnn.multiply(x, self.time_mask)  # zero padded rows so they don't skew stats
        s1 = ttnn.sum(xm, dim=1, keepdim=True)  # [1,1,1024]  sum_t x
        s2 = ttnn.sum(ttnn.multiply(xm, xm), dim=1, keepdim=True)  # [1,1,1024] sum_t x^2
        # sum over the 32 channels within each group
        s1g = ttnn.sum(ttnn.reshape(s1, (1, 1, GN_GROUPS, GN_CH)), dim=3, keepdim=True)  # [1,1,32,1]
        s2g = ttnn.sum(ttnn.reshape(s2, (1, 1, GN_GROUPS, GN_CH)), dim=3, keepdim=True)
        mu = ttnn.multiply(s1g, 1.0 / self.n)  # [1,1,32,1]
        var = ttnn.subtract(ttnn.multiply(s2g, 1.0 / self.n), ttnn.multiply(mu, mu))
        inv = ttnn.rsqrt(ttnn.add(var, 1e-5))
        # broadcast per-group stats to per-channel [1,1,1024]
        mu_c = ttnn.reshape(ttnn.repeat(mu, ttnn.Shape([1, 1, 1, GN_CH])), (1, 1, DIM))
        inv_c = ttnn.reshape(ttnn.repeat(inv, ttnn.Shape([1, 1, 1, GN_CH])), (1, 1, DIM))
        y = ttnn.multiply(ttnn.subtract(x, mu_c), inv_c)  # broadcast over S
        return ttnn.add(ttnn.multiply(y, gn_w), gn_b)

    def _attn(self, x_norm, block):
        S = self.s_pad
        qkv = ttnn.linear(x_norm, block["qkv_w"], bias=block["qkv_b"], compute_kernel_config=self.cc)  # [1,S,3072]
        qkv = ttnn.reshape(qkv, (1, S, ENC_HEADS, 3 * ENC_DH))  # heads interleaved: [q64,k64,v64] per head

        def head(sl0, sl1):
            t = qkv[:, :, :, sl0:sl1]  # [1,S,16,64]
            return ttnn.permute(t, (0, 2, 1, 3))  # [1,16,S,64]

        q = head(0, ENC_DH)
        k = head(ENC_DH, 2 * ENC_DH)
        v = head(2 * ENC_DH, 3 * ENC_DH)
        kt = ttnn.permute(k, (0, 1, 3, 2))  # [1,16,64,S]
        sim = ttnn.matmul(q, kt, compute_kernel_config=self.cc)  # [1,16,S,S]
        sim = ttnn.multiply(sim, self.scale)
        sim = ttnn.add(sim, self.key_mask)  # mask padded keys
        attn = ttnn.softmax(sim, dim=-1, compute_kernel_config=self.cc)
        a = ttnn.matmul(attn, v, compute_kernel_config=self.cc)  # [1,16,S,64]
        a = ttnn.permute(a, (0, 2, 1, 3))  # [1,S,16,64]
        a = ttnn.reshape(a, (1, S, DIM))
        return ttnn.linear(a, block["proj_w"], bias=block["proj_b"], compute_kernel_config=self.cc)

    def __call__(self, mel):  # mel [1, S, 80] -> [1, S, 1024]
        x = ttnn.linear(mel, self.p["init_w"], bias=self.p["init_b"], compute_kernel_config=self.cc)  # [1,S,1024]
        for block in self.p["blocks"]:
            x_norm = self._group_norm(x, block["gn_w"], block["gn_b"])
            x = ttnn.add(x_norm, self._attn(x_norm, block))  # residual on the NORMED input
        return x


class TTNNPerceiver:
    def __init__(self, device, parameters):
        self.device = device
        self.p = parameters
        self.cc = _compute_config()
        self.scale = DH**-0.5

    def _to_heads(self, x, n):  # [1, n, INNER] -> [1, HEADS, n, DH]
        x = ttnn.reshape(x, (1, n, HEADS, DH))
        return ttnn.permute(x, (0, 2, 1, 3))

    def _attn(self, latents, context, ctx_len, key_mask, layer):
        q = ttnn.linear(latents, layer["to_q"], compute_kernel_config=self.cc)  # [1,32,512]
        kv = ttnn.linear(context, layer["to_kv"], compute_kernel_config=self.cc)  # [1,ctx,1024]
        k = kv[:, :, 0:INNER]
        v = kv[:, :, INNER : 2 * INNER]
        qh = self._to_heads(q, LATENTS)  # [1,8,32,64]
        kh = self._to_heads(k, ctx_len)  # [1,8,ctx,64]
        vh = self._to_heads(v, ctx_len)
        kt = ttnn.permute(kh, (0, 1, 3, 2))  # [1,8,64,ctx]
        sim = ttnn.matmul(qh, kt, compute_kernel_config=self.cc)  # [1,8,32,ctx]
        sim = ttnn.multiply(sim, self.scale)
        sim = ttnn.add(sim, key_mask)  # mask padded keys
        attn = ttnn.softmax(sim, dim=-1, compute_kernel_config=self.cc)
        out = ttnn.matmul(attn, vh, compute_kernel_config=self.cc)  # [1,8,32,64]
        out = ttnn.permute(out, (0, 2, 1, 3))  # [1,32,8,64]
        out = ttnn.reshape(out, (1, LATENTS, INNER))
        return ttnn.linear(out, layer["to_out"], compute_kernel_config=self.cc)  # [1,32,1024]

    def _ff(self, x, layer):
        h = ttnn.linear(x, layer["ff1_w"], bias=layer["ff1_b"], compute_kernel_config=self.cc)  # [1,32,5460]
        a = h[:, :, 0 : h.shape[-1] // 2]
        g = h[:, :, h.shape[-1] // 2 :]
        h = ttnn.multiply(a, ttnn.gelu(g))  # GEGLU
        return ttnn.linear(h, layer["ff2_w"], bias=layer["ff2_b"], compute_kernel_config=self.cc)  # [1,32,1024]

    def __call__(self, frames, key_mask):
        """frames: [1, S_pad, 1024] (padded to tile mult); key_mask: additive [1,1,1,32+S_pad]
        (0 for valid latent+real-frame keys, -inf for padded frames). Returns [1,32,1024]."""
        latents = self.p["latents"]
        ctx_len = LATENTS + frames.shape[1]
        for layer in self.p["layers"]:
            context = ttnn.concat([latents, frames], dim=1)  # [1, 32+S_pad, 1024]
            latents = ttnn.add(latents, self._attn(latents, context, ctx_len, key_mask, layer))
            latents = ttnn.add(latents, self._ff(latents, layer))
        # RMSNorm: normalize(x,dim=-1)*sqrt(1024)*gamma == rms_norm(x)*gamma
        latents = ttnn.rms_norm(latents, weight=self.p["norm_gamma"], epsilon=1e-12, compute_kernel_config=self.cc)
        return latents

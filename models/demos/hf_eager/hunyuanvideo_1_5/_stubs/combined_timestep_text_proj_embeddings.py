# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `combined_timestep_text_proj_embeddings` of tencent/HunyuanVideo-1.5.

Reference submodule: `context_embedder.time_text_embed`, a diffusers
`CombinedTimestepTextProjEmbeddings(embedding_dim, pooled_projection_dim)`:

    timesteps_proj = self.time_proj(timestep)                       # sinusoidal (N, 256)
    timesteps_emb  = self.timestep_embedder(timesteps_proj)         # MLP 256->D->D
    pooled_proj    = self.text_embedder(pooled_projection)          # MLP P->D->D
    conditioning   = timesteps_emb + pooled_proj                    # (N, D)

Sub-modules:
    time_proj         : Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)  (weightless)
    timestep_embedder : TimestepEmbedding(256, D): linear_1 -> SiLU -> linear_2
    text_embedder     : PixArtAlphaTextProjection(P, D, act="silu"): linear_1 -> SiLU -> linear_2

Inputs at test time:
    timestep          : (N,)  1-D — PRIMARY (arrives as a ttnn tensor)
    pooled_projection : (N, P)     — arrives as a torch tensor

Native ttnn strategy
--------------------
The sinusoidal `time_proj` is weightless: its frequency vector
`exp(-ln(max_period) * arange(half)/(half - shift))` is a constant, precomputed
on host at build time and uploaded once. On device the embedding is
`args = timestep[:, None] @ freq[None, :]` (an outer product done as a matmul,
robust for any N), then `concat([cos(args), sin(args)])` (flip_sin_to_cos=True).
Both MLPs are `matmul + bias + SiLU + matmul + bias`; the two branches are summed.
All device math is float32 with a HiFi4 config. The test builds `timestep=1.0`
(bf16-exact) so the primary's bf16 cast is lossless.
"""

from __future__ import annotations

import math

import ttnn

HF_MODEL_ID = "tencent/HunyuanVideo-1.5"

# Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1).
_NUM_CHANNELS = 256
_FLIP_SIN_TO_COS = True
_DOWNSCALE_FREQ_SHIFT = 0.0
_SCALE = 1.0
_MAX_PERIOD = 10000


def build(device, torch_module):
    """Bind the two MLPs + sinusoidal frequency constant; return a native forward."""
    import torch

    m = torch_module
    te = m.timestep_embedder  # TimestepEmbedding: linear_1(256->D), act(silu), linear_2(D->D)
    tx = m.text_embedder  # PixArtAlphaTextProjection: linear_1(P->D), act_1(silu), linear_2(D->D)
    tp = m.time_proj  # Timesteps

    num_channels = int(getattr(tp, "num_channels", _NUM_CHANNELS))
    flip = bool(getattr(tp, "flip_sin_to_cos", _FLIP_SIN_TO_COS))
    shift = float(getattr(tp, "downscale_freq_shift", _DOWNSCALE_FREQ_SHIFT))
    scale = float(getattr(tp, "scale", _SCALE))
    half_dim = num_channels // 2

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def f32(t):
        return ttnn.from_torch(t.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def lin_weights(linear):
        w = f32(linear.weight.detach().t())  # (in, out)
        b = f32(linear.bias.detach().reshape(1, -1)) if linear.bias is not None else None
        return w, b

    te_w1, te_b1 = lin_weights(te.linear_1)
    te_w2, te_b2 = lin_weights(te.linear_2)
    tx_w1, tx_b1 = lin_weights(tx.linear_1)
    tx_w2, tx_b2 = lin_weights(tx.linear_2)

    # Sinusoidal frequency vector (weightless constant), as a (1, half_dim) row.
    exponent = -math.log(_MAX_PERIOD) * torch.arange(0, half_dim, dtype=torch.float32)
    exponent = exponent / (half_dim - shift)
    freq = torch.exp(exponent) * scale  # (half_dim,)
    freq_row = f32(freq.reshape(1, half_dim))  # (1, half_dim)

    def _to_f32_device(t):
        if isinstance(t, ttnn.Tensor):
            if t.get_dtype() != ttnn.float32:
                t = ttnn.typecast(t, ttnn.float32)
            return t
        return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def _mlp(x, w1, b1, w2, b2):
        h = ttnn.matmul(x, w1, compute_kernel_config=compute_config)
        if b1 is not None:
            h = ttnn.add(h, b1)
        h = ttnn.silu(h)
        h = ttnn.matmul(h, w2, compute_kernel_config=compute_config)
        if b2 is not None:
            h = ttnn.add(h, b2)
        return h

    def forward(x, pooled_projection=None, *args, **kwargs):
        if pooled_projection is None:
            if "pooled_projection" in kwargs:
                pooled_projection = kwargs["pooled_projection"]
            elif args:
                pooled_projection = args[0]
        if pooled_projection is None:
            raise TypeError("combined_timestep_text_proj_embeddings needs `pooled_projection`")

        # timestep -> (N, 1)
        ts = _to_f32_device(x)
        n = 1
        for d in ts.shape:
            n *= int(d)
        ts = ttnn.reshape(ts, (n, 1))

        # sinusoidal time projection: args = ts (N,1) @ freq (1, half) -> (N, half)
        a = ttnn.matmul(ts, freq_row, compute_kernel_config=compute_config)
        cos = ttnn.cos(a)
        sin = ttnn.sin(a)
        # flip_sin_to_cos=True -> [cos, sin]; else [sin, cos]
        proj = ttnn.concat([cos, sin] if flip else [sin, cos], dim=-1)  # (N, 256)

        timesteps_emb = _mlp(proj, te_w1, te_b1, te_w2, te_b2)  # (N, D)

        pooled = _to_f32_device(pooled_projection)
        pooled_emb = _mlp(pooled, tx_w1, tx_b1, tx_w2, tx_b2)  # (N, D)

        return ttnn.add(timesteps_emb, pooled_emb)

    return forward


def combined_timestep_text_proj_embeddings(*args, **kwargs):
    raise RuntimeError(
        "combined_timestep_text_proj_embeddings requires build(device, torch_module) "
        "to bind the MLP weights; the bare callable has no parameters."
    )

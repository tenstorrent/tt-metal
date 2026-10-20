# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-side pieces of the TRELLIS.2 structured-latent (SLat) flow DiTs: shape and texture.

The reference is `trellis2/models/structured_latent_flow.py` (microsoft/TRELLIS.2 75fbf018). The block is the one
the structure DiT uses (30 blocks, D 1536, 12 heads, ctx 1024, shared modulation), so the timestep MLP, adaLN and
the block mirror come from `structure_flow_host`. What differs: one token per active voxel (a variable count N),
RoPE on each voxel's coordinates, and 32 (shape) or 64 (texture) input channels.

Tokens are padded from N to a fixed bucket length (DESIGN D03/D07). Padded rows must never reach a real token:
self-attention is block-diagonal over [0, N) and [N, bucket), which the device gets from SDPA's
`cu_window_seqlens = [0, N, bucket]` and the mirror from an explicit mask. Every other op acts per token.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.experimental.trellis2.reference import structure_flow_host as ss

SHAPE_CKPT = "TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors"
TEX_CKPT = "TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors"
OUT_CHANNELS = 32
BUCKET_MULTIPLE = 512  # the windowed SDPA's largest key chunk (Precision.sdpa_large_k_chunk); fewer distinct buckets


def bucket_len(n: int, multiple: int = BUCKET_MULTIPLE) -> int:
    return -(-n // multiple) * multiple


def window_bounds(n: int, n_pad: int) -> list[int]:
    """cu_window_seqlens for SDPA: real tokens and padding in separate windows. Always 3 entries (the second
    window is empty when n == n_pad, which the kernel skips) so the SDPA program-cache key does not depend on N."""
    return [0, n, n_pad]


def self_attn_mask(n: int, n_pad: int) -> torch.Tensor | None:
    """[1, 1, n_pad, n_pad] additive mask equal to window_bounds(n, n_pad); None when there is no padding."""
    if n == n_pad:
        return None
    real = torch.arange(n_pad) < n
    m = torch.full((n_pad, n_pad), ss.MASK_VALUE)
    m[real[:, None] == real[None, :]] = 0.0
    return m[None, None]


def rope_tables(coords: torch.Tensor, n_pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos, sin [n_pad, head_dim] from voxel coords [N, 4] (batch, x, y, z); padded rows get phase 0."""
    cos, sin = ss.rope_tables(coords[:, 1:].long())
    pad = n_pad - coords.shape[0]
    return F.pad(cos, (0, 0, 0, pad), value=1.0), F.pad(sin, (0, 0, 0, pad))


def feats_to_tokens(feats: torch.Tensor, n_pad: int) -> torch.Tensor:
    """[N, C] -> [1, 1, n_pad, C] with zero rows appended."""
    return F.pad(feats, (0, 0, 0, n_pad - feats.shape[0]))[None, None]


def functional_forward(
    state: dict[str, torch.Tensor],
    feats: torch.Tensor,
    coords: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor,
    n_pad: int | None = None,
) -> torch.Tensor:
    """Torch mirror of the TT SLat DiT, in fp32: feats [N, C_in] (texture: noise and shape latent already
    concatenated), coords [N, 4], t [1] (x1000), cond [1, L, 1024] -> [N, 32]. n_pad defaults to the bucket."""
    n = feats.shape[0]
    n_pad = n_pad or bucket_len(n)
    s = {k: v.float() for k, v in state.items()}
    mods = ss.modulation(state, t)
    cos, sin = rope_tables(coords, n_pad)
    ctx = ss.pad_context(cond.float())
    cross = ss.cross_mask(n_pad, cond.shape[1])

    h = F.linear(feats_to_tokens(feats.float(), n_pad), s["input_layer.weight"], s["input_layer.bias"])
    h = ss.blocks_forward(s, mods, h, cos, sin, ctx, cross, self_attn_mask(n, n_pad))
    h = F.layer_norm(h, (ss.MODEL_CHANNELS,), eps=ss.FINAL_NORM_EPS)
    return F.linear(h, s["out_layer.weight"], s["out_layer.bias"])[0, 0, :n]


@dataclass
class SamplerParams:
    """FlowEulerGuidanceIntervalSampler settings (TRELLIS.2-4B pipeline.json); defaults are the shape stage."""

    steps: int = 12
    sigma_min: float = 1e-5
    guidance_strength: float = 7.5
    guidance_rescale: float = 0.5
    guidance_interval: tuple[float, float] = (0.6, 1.0)
    rescale_t: float = 3.0


TEX_SAMPLER = SamplerParams(guidance_strength=1.0, guidance_rescale=0.0, guidance_interval=(0.6, 0.9))


def _std(x: torch.Tensor) -> torch.Tensor:
    """VarLenTensor.std of one sample: population std over every element, as sqrt(E[x^2] - E[x]^2)."""
    return (x.pow(2).mean() - x.mean().pow(2)).sqrt()


def sample(model_fn, noise: torch.Tensor, cond: torch.Tensor, neg_cond: torch.Tensor, p: SamplerParams):
    """Euler flow sampling with CFG, CFG rescale and a guidance interval over sparse feats [N, C].
    model_fn(x_t, t_x1000, cond) -> v [N, C]. Returns the final sample and every model output in call order."""
    x, outs = noise.float(), []
    for t, t_prev in ss.t_pairs(p):
        tt = torch.tensor([1000 * t], dtype=torch.float32)
        cfg = p.guidance_interval[0] <= t <= p.guidance_interval[1] and p.guidance_strength != 1
        pos = model_fn(x, tt, cond).float()
        outs.append(pos)
        if cfg:
            neg = model_fn(x, tt, neg_cond).float()
            outs.append(neg)
            v = p.guidance_strength * pos + (1 - p.guidance_strength) * neg
            if p.guidance_rescale > 0:
                a, b = 1 - p.sigma_min, p.sigma_min + (1 - p.sigma_min) * t
                x0_pos, x0_cfg = a * x - b * pos, a * x - b * v
                x0_res = x0_cfg * (_std(x0_pos) / _std(x0_cfg))
                x0 = p.guidance_rescale * x0_res + (1 - p.guidance_rescale) * x0_cfg
                v = (a * x - x0) / b
        else:
            v = pos
        x = x - (t - t_prev) * v
    return x, outs

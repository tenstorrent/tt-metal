# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-side pieces of the TRELLIS.2 sparse-structure flow DiT, shared by the TT model and its torch mirror.

The reference is `trellis2/models/sparse_structure_flow.py` (microsoft/TRELLIS.2 75fbf018). The timestep MLP and
the shared adaLN projection act on one vector per call, so they stay on the host in fp32. Everything else here
builds constant tables: the 3D RoPE cos/sin, the pair-rotation matrix, and the cross-attention key mask.

`functional_forward` is a torch re-statement of the TT decomposition (per-head RMSNorm as rms_norm * gamma, RoPE
as x * cos + (x @ R) * sin, cross-attention keys padded to a tile multiple and masked). Run in fp32 against the
reference model it checks that decomposition without a device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

RESOLUTION = 16
IN_CHANNELS = 8
MODEL_CHANNELS = 1536
COND_CHANNELS = 1024
NUM_BLOCKS = 30
NUM_HEADS = 12
HEAD_DIM = MODEL_CHANNELS // NUM_HEADS
MLP_CHANNELS = int(MODEL_CHANNELS * 5.3334)
FREQ_EMBED = 256
ROPE_FREQ = (1.0, 10000.0)
CHANNEL_PAD = 32
KV_PAD_MULTIPLE = 128
BLOCK_NORM_EPS = 1e-6
FINAL_NORM_EPS = 1e-5
QK_NORM_EPS = 1e-6
MASK_VALUE = -1e9

HOST_KEYS = ("t_embedder.", "adaLN_modulation.")


def load_state_dict(path: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(path)


def timestep_embedding(t: torch.Tensor, dim: int = FREQ_EMBED, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


@dataclass
class Modulation:
    """Per-block adaLN vectors for one call, each [NUM_BLOCKS, D] in fp32; scales are stored as 1 + scale."""

    shift_msa: torch.Tensor
    scale_msa: torch.Tensor
    gate_msa: torch.Tensor
    shift_mlp: torch.Tensor
    scale_mlp: torch.Tensor
    gate_mlp: torch.Tensor


def modulation(state: dict[str, torch.Tensor], t: torch.Tensor) -> Modulation:
    """t is the model's timestep input (already multiplied by 1000 by the sampler), shape [1]."""
    w = {k: v.float() for k, v in state.items() if k.startswith(HOST_KEYS)}
    h = timestep_embedding(t)
    h = F.linear(h, w["t_embedder.mlp.0.weight"], w["t_embedder.mlp.0.bias"])
    h = F.linear(F.silu(h), w["t_embedder.mlp.2.weight"], w["t_embedder.mlp.2.bias"])
    mod = F.linear(F.silu(h), w["adaLN_modulation.1.weight"], w["adaLN_modulation.1.bias"])
    per_block = torch.stack([state[f"blocks.{i}.modulation"].float() for i in range(NUM_BLOCKS)]) + mod
    parts = per_block.reshape(NUM_BLOCKS, 6, MODEL_CHANNELS).unbind(1)
    return Modulation(parts[0], 1 + parts[1], parts[2], parts[3], 1 + parts[4], parts[5])


def grid_coords(resolution: int = RESOLUTION) -> torch.Tensor:
    axes = [torch.arange(resolution)] * 3
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)


def rope_tables(coords: torch.Tensor, head_dim: int = HEAD_DIM) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin, [N, head_dim], for rotation of adjacent pairs (x[2j], x[2j+1]) as TRELLIS.2 does with
    view_as_complex. Each axis gets head_dim // 6 frequencies; the leftover pairs get phase 0 (identity)."""
    freq_dim = head_dim // 2 // 3
    freqs = ROPE_FREQ[0] / (ROPE_FREQ[1] ** (torch.arange(freq_dim, dtype=torch.float32) / freq_dim))
    phases = (coords.float()[:, :, None] * freqs[None, None]).reshape(coords.shape[0], -1)
    phases = F.pad(phases, (0, head_dim // 2 - phases.shape[1]))
    return torch.cos(phases).repeat_interleave(2, -1), torch.sin(phases).repeat_interleave(2, -1)


def pair_rotation(head_dim: int = HEAD_DIM) -> torch.Tensor:
    """R with (x @ R)[2j] = -x[2j+1] and (x @ R)[2j+1] = x[2j]."""
    r = torch.zeros(head_dim, head_dim)
    idx = torch.arange(0, head_dim, 2)
    r[idx + 1, idx] = -1.0
    r[idx, idx + 1] = 1.0
    return r


def padded_kv_len(n: int) -> int:
    return -(-n // KV_PAD_MULTIPLE) * KV_PAD_MULTIPLE


def pad_context(cond: torch.Tensor) -> torch.Tensor:
    """[1, L, C] -> [1, 1, Lp, C], zero rows appended up to a multiple of KV_PAD_MULTIPLE."""
    n = cond.shape[1]
    return F.pad(cond, (0, 0, 0, padded_kv_len(n) - n)).unsqueeze(0)


def cross_mask(n_q: int, n_kv: int) -> torch.Tensor:
    """[1, 1, n_q, padded] additive mask: 0 on the n_kv real keys, MASK_VALUE on the padding."""
    m = torch.zeros(1, 1, n_q, padded_kv_len(n_kv))
    m[..., n_kv:] = MASK_VALUE
    return m


def x_to_tokens(x: torch.Tensor) -> torch.Tensor:
    """[1, C, R, R, R] -> [1, 1, N, CHANNEL_PAD] (channels zero-padded to a tile)."""
    h = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    return F.pad(h, (0, CHANNEL_PAD - h.shape[-1])).unsqueeze(0)


def tokens_to_x(h: torch.Tensor, channels: int = IN_CHANNELS, resolution: int = RESOLUTION) -> torch.Tensor:
    """[1, 1, N, >= C] -> [1, C, R, R, R]."""
    h = h.reshape(1, -1, h.shape[-1])[..., :channels]
    return h.permute(0, 2, 1).reshape(1, channels, resolution, resolution, resolution).contiguous()


def _heads(x: torch.Tensor) -> torch.Tensor:
    b, _, n, d = x.shape
    return x.reshape(b, n, NUM_HEADS, d // NUM_HEADS).permute(0, 2, 1, 3)


def _merge(x: torch.Tensor) -> torch.Tensor:
    b, h, n, e = x.shape
    return x.permute(0, 2, 1, 3).reshape(b, 1, n, h * e)


def _rms(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + QK_NORM_EPS) * gamma[None, :, None, :]


@dataclass
class SamplerParams:
    """FlowEulerGuidanceIntervalSampler settings of the structure stage (TRELLIS.2-4B pipeline.json)."""

    steps: int = 12
    sigma_min: float = 1e-5
    guidance_strength: float = 7.5
    guidance_rescale: float = 0.7
    guidance_interval: tuple[float, float] = (0.6, 1.0)
    rescale_t: float = 5.0


def t_pairs(p: SamplerParams) -> list[tuple[float, float]]:
    seq = [i / p.steps for i in range(p.steps, -1, -1)]
    seq = [p.rescale_t * s / (1 + (p.rescale_t - 1) * s) for s in seq]
    return list(zip(seq[:-1], seq[1:]))


def sample(
    model_fn, noise: torch.Tensor, cond: torch.Tensor, neg_cond: torch.Tensor, p: SamplerParams = SamplerParams()
):
    """Euler flow sampling with CFG, CFG rescale and a guidance interval, as TRELLIS.2's sampler mixins do.
    model_fn(x_t, t_x1000, cond) -> v. Returns the final sample and every model output in call order."""
    x, outs = noise.float(), []
    for t, t_prev in t_pairs(p):
        tt = torch.tensor([1000 * t], dtype=torch.float32)
        pos = model_fn(x, tt, cond).float()
        outs.append(pos)
        if p.guidance_interval[0] <= t <= p.guidance_interval[1]:
            neg = model_fn(x, tt, neg_cond).float()
            outs.append(neg)
            v = p.guidance_strength * pos + (1 - p.guidance_strength) * neg
            if p.guidance_rescale > 0:
                a, b = 1 - p.sigma_min, p.sigma_min + (1 - p.sigma_min) * t
                x0_pos, x0_cfg = a * x - b * pos, a * x - b * v
                dims = list(range(1, x.ndim))
                x0_res = x0_cfg * (x0_pos.std(dim=dims, keepdim=True) / x0_cfg.std(dim=dims, keepdim=True))
                x0 = p.guidance_rescale * x0_res + (1 - p.guidance_rescale) * x0_cfg
                v = (a * x - x0) / b
        else:
            v = pos
        x = x - (t - t_prev) * v
    return x, outs


def functional_forward(state: dict[str, torch.Tensor], x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
    """Torch mirror of the TT model, in fp32. Same inputs and output as SparseStructureFlowModel.forward."""
    s = {k: v.float() for k, v in state.items()}
    mods = modulation(state, t)
    cos, sin = rope_tables(grid_coords())
    rot = pair_rotation()
    ctx = pad_context(cond.float())
    mask = cross_mask(RESOLUTION**3, cond.shape[1])

    w_in = F.pad(s["input_layer.weight"], (0, CHANNEL_PAD - IN_CHANNELS))
    h = F.linear(x_to_tokens(x.float()), w_in, s["input_layer.bias"])
    for i in range(NUM_BLOCKS):
        p = f"blocks.{i}."
        q_w, k_w, v_w = s[p + "self_attn.to_qkv.weight"].chunk(3)
        q_b, k_b, v_b = s[p + "self_attn.to_qkv.bias"].chunk(3)

        a = F.layer_norm(h, (MODEL_CHANNELS,), eps=BLOCK_NORM_EPS) * mods.scale_msa[i] + mods.shift_msa[i]
        q = _rms(_heads(F.linear(a, q_w, q_b)), s[p + "self_attn.q_rms_norm.gamma"])
        k = _rms(_heads(F.linear(a, k_w, k_b)), s[p + "self_attn.k_rms_norm.gamma"])
        v = _heads(F.linear(a, v_w, v_b))
        q = q * cos + (q @ rot) * sin
        k = k * cos + (k @ rot) * sin
        a = _merge(F.scaled_dot_product_attention(q, k, v))
        h = h + F.linear(a, s[p + "self_attn.to_out.weight"], s[p + "self_attn.to_out.bias"]) * mods.gate_msa[i]

        kv_w, kv_b = s[p + "cross_attn.to_kv.weight"], s[p + "cross_attn.to_kv.bias"]
        a = F.layer_norm(h, (MODEL_CHANNELS,), s[p + "norm2.weight"], s[p + "norm2.bias"], eps=BLOCK_NORM_EPS)
        q = _rms(
            _heads(F.linear(a, s[p + "cross_attn.to_q.weight"], s[p + "cross_attn.to_q.bias"])),
            s[p + "cross_attn.q_rms_norm.gamma"],
        )
        k = _rms(
            _heads(F.linear(ctx, kv_w[:MODEL_CHANNELS], kv_b[:MODEL_CHANNELS])), s[p + "cross_attn.k_rms_norm.gamma"]
        )
        v = _heads(F.linear(ctx, kv_w[MODEL_CHANNELS:], kv_b[MODEL_CHANNELS:]))
        a = _merge(F.scaled_dot_product_attention(q, k, v, attn_mask=mask))
        h = h + F.linear(a, s[p + "cross_attn.to_out.weight"], s[p + "cross_attn.to_out.bias"])

        a = F.layer_norm(h, (MODEL_CHANNELS,), eps=BLOCK_NORM_EPS) * mods.scale_mlp[i] + mods.shift_mlp[i]
        a = F.gelu(F.linear(a, s[p + "mlp.mlp.0.weight"], s[p + "mlp.mlp.0.bias"]), approximate="tanh")
        h = h + F.linear(a, s[p + "mlp.mlp.2.weight"], s[p + "mlp.mlp.2.bias"]) * mods.gate_mlp[i]

    h = F.layer_norm(h, (MODEL_CHANNELS,), eps=FINAL_NORM_EPS)
    w_out = F.pad(s["out_layer.weight"], (0, 0, 0, CHANNEL_PAD - IN_CHANNELS))
    b_out = F.pad(s["out_layer.bias"], (0, CHANNEL_PAD - IN_CHANNELS))
    return tokens_to_x(F.linear(h, w_out, b_out))

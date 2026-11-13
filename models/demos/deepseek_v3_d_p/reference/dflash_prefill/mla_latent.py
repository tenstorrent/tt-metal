# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Torch reference for the MLA (DFlash2) drafter's prefill context path.

The vendored HF drafter (``dflash.py``) is the GQA Qwen3 variant and cannot be the golden for an MLA drafter.
This is a port of tt-blaze's ``tests/blaze/dflash/mla_golden.py`` (commit 8b7c7cbafa, branch aling/k3-spec-mla) --
itself a port of TorchSpec's ``K3DSparkMLAAttention`` -- restricted to what prefill builds:

    context_feature = hidden_norm(fc(concat(taps)))
    latent_kv       = [ kv_a_layernorm(c_kv) | rope_interleaved(k_pe) ]   per draft layer   (N, 576)

Keep :func:`yarn_inv_freq`, :func:`rope_tables`, :func:`rope_interleaved` and :func:`latent_kv` line-for-line
with blaze's: the decode side reads what this produces, so the two references must agree exactly.
"""

from __future__ import annotations

import math

import torch

from models.demos.deepseek_v3_d_p.tt.dflash_prefill.dflash_drafter_config import DFlashDrafterConfig

LAYER_PREFIX = "layers.{layer}."


def _yarn_get_mscale(scale: float, mscale: float) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_inv_freq(cfg: DFlashDrafterConfig) -> tuple[torch.Tensor, float]:
    """``LlamaYarnRotaryEmbedding`` (TorchSpec ``llama3_eagle.py``): ``(inv_freq, cos/sin scale)``."""
    dim, base = cfg.qk_rope_head_dim, cfg.rope_theta
    freq_extra = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freq_inter = 1.0 / (cfg.rope_factor * base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    def correction_dim(num_rotations):
        return (dim * math.log(cfg.rope_orig_max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(correction_dim(cfg.rope_beta_fast)), 0)
    high = min(math.ceil(correction_dim(cfg.rope_beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    inv_freq_mask = 1.0 - ramp
    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
    scale = _yarn_get_mscale(cfg.rope_factor, cfg.rope_mscale) / _yarn_get_mscale(
        cfg.rope_factor, cfg.rope_mscale_all_dim
    )
    return inv_freq, float(scale)


def rope_tables(positions: torch.Tensor, cfg: DFlashDrafterConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved-pair cos/sin rows ``(N, rope_dim)``: pair ``i`` repeats frequency ``i`` twice."""
    inv_freq, scale = yarn_inv_freq(cfg)
    freqs = torch.outer(positions.to(torch.float32), inv_freq)
    cos = torch.repeat_interleave(freqs.cos() * scale, 2, dim=-1)
    sin = torch.repeat_interleave(freqs.sin() * scale, 2, dim=-1)
    return cos, sin


def rope_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x * cos + rotate_half_interleaved(x) * sin`` (TorchSpec ``_apply_rope_interleaved``)."""
    x = x.float()
    rot = torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)
    return x * cos.float() + rot * sin.float()


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()


def context_feature(sd: dict, taps: list[torch.Tensor], cfg: DFlashDrafterConfig) -> torch.Tensor:
    """``hidden_norm(fc(concat(taps)))`` ``(N, hidden)``; ``taps`` in ``cfg.target_layer_ids`` order."""
    x = torch.cat([t.float() for t in taps], dim=-1)
    return rmsnorm(x @ sd["fc.weight"].float().T, sd["hidden_norm.weight"], cfg.rms_norm_eps)


def latent_kv(sd: dict, layer: int, x: torch.Tensor, positions: torch.Tensor, cfg: DFlashDrafterConfig) -> torch.Tensor:
    """Cached latent rows ``(N, 576)``: ``[kv_a_layernorm(c_kv) | rope(k_pe)]`` of ``x @ kv_a`` (blaze ``latent_kv``)."""
    p = LAYER_PREFIX.format(layer=layer)
    kv = x.float() @ sd[p + "self_attn.kv_a_proj_with_mqa.weight"].float().T
    c_kv, k_pe = kv.split([cfg.kv_lora_rank, cfg.qk_rope_head_dim], dim=-1)
    cos, sin = rope_tables(positions, cfg)
    c_kv = rmsnorm(c_kv, sd[p + "self_attn.kv_a_layernorm.weight"], cfg.rms_norm_eps)
    return torch.cat([c_kv, rope_interleaved(k_pe, cos, sin)], dim=-1)


def context_latents(sd: dict, taps: list[torch.Tensor], cfg: DFlashDrafterConfig, start: int = 0) -> torch.Tensor:
    """Every draft layer's latent for context positions ``[start, start + N)``: ``(layers, N, 576)``."""
    x = context_feature(sd, taps, cfg)
    positions = torch.arange(start, start + x.shape[0])
    return torch.stack([latent_kv(sd, i, x, positions, cfg) for i in range(cfg.num_hidden_layers)])


def synthetic_prefill_state_dict(cfg: DFlashDrafterConfig, *, seed: int = 0, scale: float = 0.02) -> dict:
    """Checkpoint-convention random weights for the prefill subset (bf16): fc, hidden_norm and per-layer
    kv_a_proj_with_mqa / kv_a_layernorm. Matrices at ``scale``; gammas ``1 + 0.1 N`` (as blaze's synthetic dict)."""
    g = torch.Generator().manual_seed(seed)
    H, n_taps = cfg.hidden_size, len(cfg.target_layer_ids)

    def mat(o, i):
        return (torch.randn(o, i, generator=g) * scale).to(torch.bfloat16)

    def gamma(n):
        return (1.0 + 0.1 * torch.randn(n, generator=g)).to(torch.bfloat16)

    sd = {"fc.weight": mat(H, n_taps * H), "hidden_norm.weight": gamma(H)}
    for i in range(cfg.num_hidden_layers):
        p = LAYER_PREFIX.format(layer=i)
        sd[p + "self_attn.kv_a_proj_with_mqa.weight"] = mat(cfg.latent_dim, H)
        sd[p + "self_attn.kv_a_layernorm.weight"] = gamma(cfg.kv_lora_rank)
    return sd


__all__ = [
    "context_feature",
    "context_latents",
    "latent_kv",
    "rope_interleaved",
    "rope_tables",
    "synthetic_prefill_state_dict",
    "yarn_inv_freq",
]

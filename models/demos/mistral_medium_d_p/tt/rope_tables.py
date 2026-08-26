# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 YaRN table math — PURE TORCH, no ttnn import.

Kept free of ttnn so the host-only conformance test (tests/unit/test_rope_vs_hf.py) runs on a
dev box with no TT runtime built. ``tt/rope.py`` re-exports everything here and adds the
device-side builders.

FULL rotary (rotary_dim == head_dim == 128) with YaRN scaling.

Ported from ``gpt_oss_d_p/tt/rope.py``, with three corrections for this checkpoint:

1. **truncate=True.** gpt-oss sets ``rope_scaling.truncate=False`` and therefore keeps the YaRN
   correction dims as floats. Mistral's config has no ``truncate`` key, so HF's default (``True``)
   applies and the correction range is floor/ceil'd. Getting this backwards shifts the
   interpolation<->extrapolation ramp and injects a per-frequency phase error that grows linearly
   with position — invisible at short seq, fatal at 128K.
2. **mscale / mscale_all_dim.** HF (``_compute_yarn_parameters``) only takes the DeepSeek
   ``m(f,mscale)/m(f,mscale_all_dim)`` ratio when BOTH are truthy. Mistral ships
   ``mscale=1.0, mscale_all_dim=0.0`` — ``0.0`` is falsy — so HF falls back to ``m(factor)``.
   Both branches are implemented here; for this config they agree at 1.4158883.
3. **beta_fast=4.0 / beta_slow=1.0** (gpt-oss uses 32/1) and theta 1e6, factor 64 over orig ctx 4096.

The attention_factor multiplies BOTH cos and sin, i.e. q and k are each scaled by ~1.4159 and the
scores run ~2.0x hot. That is the YaRN temperature; it is free here because it is baked into the
host-built table, but dropping it is silently wrong.

Tables are emitted in the **Meta interleaved** convention ``[c0, c0, c1, c1, ...]`` — what
``ttnn.experimental.rotary_embedding_llama`` / ``rotary_embedding_indexed`` +
``get_rot_transformation_mat`` expect, and what pairs with ``convert_hf_qkv_to_meta_format``-swizzled
q/k projections.
"""

import math

import torch

# Mistral-Medium-3.5-128B YaRN parameters (configs/Mistral-Medium-3.5-128B/config.json).
DEFAULT_ROPE_THETA = 1000000.0
DEFAULT_YARN_FACTOR = 64.0
DEFAULT_YARN_ORIG_MAX_POS = 4096
DEFAULT_YARN_BETA_FAST = 4.0
DEFAULT_YARN_BETA_SLOW = 1.0
DEFAULT_YARN_MSCALE = 1.0
DEFAULT_YARN_MSCALE_ALL_DIM = 0.0
DEFAULT_YARN_TRUNCATE = True


def yarn_params_from_config(hf_config):
    """Extract the YaRN kwargs from an HF config (transformers>=5 ``rope_parameters``, or the
    legacy ``rope_scaling``). Returns a kwargs dict for :func:`build_yarn_cos_sin`.

    Fails loud on a non-YaRN rope_type: silently falling through to unscaled (or, worse, llama3)
    RoPE is the single most expensive bug available here.
    """
    rp = getattr(hf_config, "rope_parameters", None) or getattr(hf_config, "rope_scaling", None) or {}
    if isinstance(rp, dict) and "full_attention" in rp:  # Gemma-style nesting; not expected here
        rp = rp["full_attention"]
    rope_type = (rp.get("rope_type") or rp.get("type") or "default").lower()
    if rope_type not in ("yarn",):
        raise ValueError(
            f"mistral_medium_d_p RoPE expects rope_type='yarn', got {rope_type!r}. "
            "Refusing to fall back to unscaled RoPE — that would be silently wrong at long context."
        )
    theta = rp.get("rope_theta") or getattr(hf_config, "rope_theta", None) or DEFAULT_ROPE_THETA
    return dict(
        rope_theta=float(theta),
        yarn_factor=float(rp["factor"]),
        yarn_orig_max_pos=int(rp["original_max_position_embeddings"]),
        yarn_beta_fast=float(rp.get("beta_fast", DEFAULT_YARN_BETA_FAST)),
        yarn_beta_slow=float(rp.get("beta_slow", DEFAULT_YARN_BETA_SLOW)),
        yarn_mscale=rp.get("mscale", DEFAULT_YARN_MSCALE),
        yarn_mscale_all_dim=rp.get("mscale_all_dim", DEFAULT_YARN_MSCALE_ALL_DIM),
        yarn_truncate=bool(rp.get("truncate", DEFAULT_YARN_TRUNCATE)),
    )


def _get_mscale(scale, mscale=1.0):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_inv_freq(
    head_dim,
    base=DEFAULT_ROPE_THETA,
    factor=DEFAULT_YARN_FACTOR,
    orig_max_pos=DEFAULT_YARN_ORIG_MAX_POS,
    beta_fast=DEFAULT_YARN_BETA_FAST,
    beta_slow=DEFAULT_YARN_BETA_SLOW,
    mscale=DEFAULT_YARN_MSCALE,
    mscale_all_dim=DEFAULT_YARN_MSCALE_ALL_DIM,
    truncate=DEFAULT_YARN_TRUNCATE,
):
    """YaRN inverse frequencies + attention_factor. Bit-for-bit ``_compute_yarn_parameters``
    (transformers/modeling_rope_utils.py) for this family; see tests/unit/test_rope_vs_hf.py."""

    def find_correction_dim(num_rotations):
        return (head_dim * math.log(orig_max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    low, high = find_correction_dim(beta_fast), find_correction_dim(beta_slow)
    if truncate:  # HF default; Mistral's config omits `truncate` so this branch is the live one
        low, high = math.floor(low), math.ceil(high)
    low = max(low, 0)
    high = min(high, head_dim - 1)

    pos_freqs = base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    if low == high:
        high += 0.001  # prevent singularity, as HF does
    ramp = ((torch.arange(head_dim // 2).float() - low) / (high - low)).clamp(0, 1)
    inv_freq_extrapolation_factor = 1.0 - ramp

    inv_freq = (
        inv_freq_interpolation * (1.0 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    # HF takes the DeepSeek ratio only when BOTH mscale fields are truthy. Mistral has
    # mscale_all_dim == 0.0 (falsy) -> plain m(factor) == 1.4158883 for factor=64.
    if mscale and mscale_all_dim:
        attention_factor = float(_get_mscale(factor, mscale) / _get_mscale(factor, mscale_all_dim))
    else:
        attention_factor = _get_mscale(factor)
    return inv_freq, attention_factor


def build_yarn_cos_sin(
    seq_len,
    head_dim,
    *,
    rope_theta=DEFAULT_ROPE_THETA,
    yarn_factor=DEFAULT_YARN_FACTOR,
    yarn_orig_max_pos=DEFAULT_YARN_ORIG_MAX_POS,
    yarn_beta_fast=DEFAULT_YARN_BETA_FAST,
    yarn_beta_slow=DEFAULT_YARN_BETA_SLOW,
    yarn_mscale=DEFAULT_YARN_MSCALE,
    yarn_mscale_all_dim=DEFAULT_YARN_MSCALE_ALL_DIM,
    yarn_truncate=DEFAULT_YARN_TRUNCATE,
    start_pos=0,
):
    """Meta interleaved cos/sin ``[1, 1, seq_len, head_dim]`` with the YaRN attention_factor folded in."""
    inv_freq, attn_factor = yarn_inv_freq(
        head_dim,
        rope_theta,
        yarn_factor,
        yarn_orig_max_pos,
        yarn_beta_fast,
        yarn_beta_slow,
        yarn_mscale,
        yarn_mscale_all_dim,
        yarn_truncate,
    )
    pos = torch.arange(start_pos, start_pos + seq_len).float()
    freqs = torch.outer(pos, inv_freq)  # [seq_len, head_dim/2]
    cos_half = torch.cos(freqs) * attn_factor
    sin_half = torch.sin(freqs) * attn_factor
    cos_meta = torch.stack([cos_half, cos_half], dim=-1).flatten(-2)[None, None]
    sin_meta = torch.stack([sin_half, sin_half], dim=-1).flatten(-2)[None, None]
    return cos_meta, sin_meta


def build_hf_cos_sin(seq_len, head_dim, **kwargs):
    """HF-convention cos/sin ``[seq_len, head_dim]`` = ``cat([half, half], -1)``, for torch
    references that use ``rotate_half``. Same frequencies/attention_factor as
    :func:`build_yarn_cos_sin`; only the layout differs."""
    start_pos = kwargs.pop("start_pos", 0)
    inv_freq, attn_factor = yarn_inv_freq(
        head_dim,
        kwargs.get("rope_theta", DEFAULT_ROPE_THETA),
        kwargs.get("yarn_factor", DEFAULT_YARN_FACTOR),
        kwargs.get("yarn_orig_max_pos", DEFAULT_YARN_ORIG_MAX_POS),
        kwargs.get("yarn_beta_fast", DEFAULT_YARN_BETA_FAST),
        kwargs.get("yarn_beta_slow", DEFAULT_YARN_BETA_SLOW),
        kwargs.get("yarn_mscale", DEFAULT_YARN_MSCALE),
        kwargs.get("yarn_mscale_all_dim", DEFAULT_YARN_MSCALE_ALL_DIM),
        kwargs.get("yarn_truncate", DEFAULT_YARN_TRUNCATE),
    )
    pos = torch.arange(start_pos, start_pos + seq_len).float()
    freqs = torch.outer(pos, inv_freq)
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return cos, sin

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma-local denoise attention.

The DiffusionGemma denoise pass is **bidirectional and read-only on the KV cache**:
canvas queries attend to a frozen prompt prefix (concatenated in front of the
canvas K/V) plus the canvas itself, and nothing is written back into the
committed cache. That write discipline is the ``DENOISE_READONLY`` phase (see
:mod:`models.experimental.diffusion_gemma.kv_phase`).

Rather than thread non-causal / prefix-KV / RoPE-offset knobs through the shared
``models.demos.gemma4`` attention op (which would touch the production causal
path), this module owns the denoise SDPA itself and **reuses Gemma4's importable
building blocks** (QKV projection, head split, per-head norm, RoPE, head concat,
output projection, all-reduce). The prefill-write and commit-append phases are
plain Gemma4 prefill/decode and need no diffusion-specific code.

It operates on a live ``Gemma4Attention`` instance, reading its ``weights`` /
``config`` / ``mesh_config`` / ``ccl_manager`` — so the decoder weights are shared
with the backbone, only the attention math differs.
"""

from __future__ import annotations

import os

import ttnn

from models.demos.gemma4.tt.attention.operations import (
    apply_allreduce,
    apply_output_projection,
    apply_per_head_norm,
    apply_qkv_projection,
    apply_rope,
    concat_heads,
    split_qkv_heads_prefill,
)

TILE_SIZE = 32


def validate_q_rope_offset(q_rope_offset: int) -> None:
    if q_rope_offset % TILE_SIZE != 0:
        raise ValueError(f"q_rope_offset must be a multiple of {TILE_SIZE}, got {q_rope_offset}")


def _largest_tile_divisor(length, preferred):
    """Pick a tile-multiple chunk size that divides ``length``."""
    start = (min(preferred, length) // TILE_SIZE) * TILE_SIZE
    for candidate in range(start, TILE_SIZE - 1, -TILE_SIZE):
        if length % candidate == 0:
            return candidate
    return TILE_SIZE


def _denoise_sdpa_program_config(head_dim, q_seq_len, k_seq_len):
    """SDPAProgramConfig for the rectangular non-causal denoise SDPA.

    Mirrors the Gemma4 prefill tuning (head_dim-dependent grid + default chunks)
    but allows ``q_seq_len != k_seq_len`` (canvas Q vs prompt+canvas K), so chunk
    sizes must independently divide each axis.
    """
    if head_dim >= 512:
        grid = ttnn.CoreCoord(8, 4)
        dq, dk = 128, 128
    else:
        grid = ttnn.CoreCoord(8, 8)
        dq, dk = 256, 128
    q_chunk = int(os.environ.get("GEMMA4_PREFILL_SDPA_QCHUNK", dq))
    k_chunk = int(os.environ.get("GEMMA4_PREFILL_SDPA_KCHUNK", dk))
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=_largest_tile_divisor(q_seq_len, q_chunk),
        k_chunk_size=_largest_tile_divisor(k_seq_len, k_chunk),
        exp_approx_mode=False,
    )


def _slice_rope_cache(cache, start, length):
    if start % TILE_SIZE != 0:
        raise ValueError(f"RoPE cache start must be a multiple of {TILE_SIZE}, got {start}")
    if start + length > cache.shape[-2]:
        raise ValueError(f"RoPE cache slice [{start}, {start + length}) exceeds cache length {cache.shape[-2]}")
    if start == 0 and cache.shape[-2] == length:
        return cache
    return ttnn.slice(cache, [0, 0, start, 0], [cache.shape[0], cache.shape[1], start + length, cache.shape[3]])


def denoise_attention(
    attn,
    hidden_states,
    *,
    rope_mats,
    attn_mask=None,
    kv_hidden_states=None,
    prefix_kv=None,
    q_rope_offset=0,
):
    """Read-only bidirectional denoise attention for one decoder layer.

    Args:
        attn: a ``Gemma4Attention`` instance (shared backbone weights/config).
        hidden_states: canvas hidden states ``[1, 1, C, H]`` (already normed by
            the caller's ``input_layernorm``).
        rope_mats: ``(cos_cache, sin_cache)`` for this layer, sliced/offset by the
            caller to cover ``q_rope_offset + C``.
        attn_mask: optional additive ``[1, 1, C, P+C]`` mask. ``None`` = all-attend.
        kv_hidden_states: optional ``[1, 1, P+C, H]`` source to recompute K/V from
            (legacy hidden-state shim); mutually exclusive with ``prefix_kv``.
        prefix_kv: optional projected frozen prompt ``(K, V)`` heads concatenated in
            front of the canvas K/V (the cache-shaped production path).
        q_rope_offset: absolute RoPE position of the first canvas token (``prompt_len``).

    Returns:
        Attention output for the canvas positions ``[1, 1, C, H]``.
    """
    if kv_hidden_states is not None and prefix_kv is not None:
        raise ValueError("pass at most one of kv_hidden_states or prefix_kv")
    validate_q_rope_offset(q_rope_offset)

    weights = attn.weights
    config = attn.config
    mesh_config = attn.mesh_config
    ccl_manager = attn.ccl_manager
    cos_cache, sin_cache = rope_mats
    tp = mesh_config.tp if mesh_config else 1

    xqkv = apply_qkv_projection(hidden_states, weights)
    tt_q, tt_k, tt_v = split_qkv_heads_prefill(
        xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
    )
    if kv_hidden_states is not None:
        xqkv_kv = apply_qkv_projection(kv_hidden_states, weights)
        tt_kv_q, tt_k_from_kv, tt_v_from_kv = split_qkv_heads_prefill(
            xqkv_kv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
        )
        tt_k.deallocate(True)
        tt_v.deallocate(True)
        tt_kv_q.deallocate(True)
        tt_k, tt_v = tt_k_from_kv, tt_v_from_kv

    tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)
    tt_k = apply_per_head_norm(tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
    tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False)

    q_cos = _slice_rope_cache(cos_cache, q_rope_offset, tt_q.shape[-2])
    q_sin = _slice_rope_cache(sin_cache, q_rope_offset, tt_q.shape[-2])
    tt_q = apply_rope(tt_q, q_cos, q_sin)
    if q_cos is not cos_cache:
        q_cos.deallocate(True)
    if q_sin is not sin_cache:
        q_sin.deallocate(True)

    # When K is just the canvas (kv_hidden recomputes the full prompt+canvas, so its
    # RoPE starts at 0), only the canvas-only K path needs the prompt_len offset.
    k_rope_offset = q_rope_offset if prefix_kv is not None else 0
    k_cos = _slice_rope_cache(cos_cache, k_rope_offset, tt_k.shape[-2])
    k_sin = _slice_rope_cache(sin_cache, k_rope_offset, tt_k.shape[-2])
    tt_k = apply_rope(tt_k, k_cos, k_sin)
    if k_cos is not cos_cache:
        k_cos.deallocate(True)
    if k_sin is not sin_cache:
        k_sin.deallocate(True)

    if prefix_kv is not None:
        prefix_k, prefix_v = prefix_kv
        canvas_k, canvas_v = tt_k, tt_v
        prefix_k_concat = ttnn.to_memory_config(prefix_k, canvas_k.memory_config())
        prefix_v_concat = ttnn.to_memory_config(prefix_v, canvas_v.memory_config())
        tt_k = ttnn.concat([prefix_k_concat, canvas_k], dim=2)
        tt_v = ttnn.concat([prefix_v_concat, canvas_v], dim=2)
        if prefix_k_concat is not prefix_k:
            prefix_k_concat.deallocate(True)
        if prefix_v_concat is not prefix_v:
            prefix_v_concat.deallocate(True)
        canvas_k.deallocate(True)
        canvas_v.deallocate(True)

    q_seq_len = tt_q.shape[-2]
    k_seq_len = tt_k.shape[-2]
    program_config = _denoise_sdpa_program_config(config.head_dim, q_seq_len, k_seq_len)
    if attn_mask is not None:
        tt_sdpa = ttnn.transformer.scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            attn_mask=attn_mask,
            is_causal=False,
            scale=1.0,
            program_config=program_config,
        )
    else:
        tt_sdpa = ttnn.transformer.scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            is_causal=False,
            scale=1.0,
            program_config=program_config,
        )
    tt_q.deallocate(True)
    tt_k.deallocate(True)
    tt_v.deallocate(True)

    tt_out = concat_heads(tt_sdpa, is_decode_mode=False)
    tt_out = apply_output_projection(tt_out, weights)
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, config.hidden_size)
    return tt_out

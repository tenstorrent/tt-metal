# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Chain all 5 DFlash drafter layers. Each layer reads its OWN persistent, already
projected+RoPE'd K/V cache for context (see attention.py's
``project_and_cache_context_delta``/``dflash_drafter_update_kv_caches`` below) and only
computes the noise/draft block's own Q/K/V live; only the noise/draft-block
representation threads from one layer to the next. Each layer's mask depends only on
(is_causal, sliding_window) -- built once per distinct combination and reused
(models/demos/gemma4/docs/dflash_design.md section 1: layers 1-4 are sliding/causal,
layer 5 is full/bidirectional)."""

from __future__ import annotations

import ttnn
from models.demos.gemma4.tt.dflash.attention import (
    _dflash_pad_noise_concat_enabled,
    _tile_pad_len,
    build_attention_mask_additive_device,
    build_attention_mask_additive_device_dynamic,
    combine_attention_mask_dynamic,
    project_and_cache_context_delta,
)
from models.demos.gemma4.tt.dflash.layer import dflash_layer_forward
from models.demos.gemma4.tt.dflash.weights import Gemma4DFlashWeights


def dflash_drafter_forward(
    kv_caches: list[tuple[ttnn.Tensor, ttnn.Tensor]],  # one (k_cache, v_cache) pair per layer
    noise: ttnn.Tensor,
    weights: Gemma4DFlashWeights,
    cos_noise: ttnn.Tensor,  # [1,1,q_len,head_dim] -- noise's own positions only, context never re-RoPE'd here
    sin_noise: ttnn.Tensor,
    mesh_device,
    mesh_config,
    ccl_manager,
    num_local_heads: int,
    num_local_kv_heads: int,
    head_dim: int,
    eps: float,
    layer_configs: list[tuple[bool, int | None]],  # (is_causal, sliding_window) per layer
    max_seq_len: int,  # each cache's fixed sequence-axis width (ctx_len for masking)
    context_valid_len_tt: ttnn.Tensor | None = None,
    mask_static_parts: dict[tuple[bool, int | None], "DynamicMaskStaticParts"] | None = None,
) -> ttnn.Tensor:
    """``context_valid_len_tt``: when given (a ``[1,1]`` int32 device tensor), each
    layer's cache (fixed ``max_seq_len``-wide) is treated as having only its first
    ``context_valid_len_tt`` rows real, the rest masked-out padding -- see attention.py's
    ``build_attention_mask_additive_device_dynamic``. This is what a growing generation
    session needs, since the REAL amount of accumulated context varies iteration to
    iteration but a trace's tensor shapes cannot. When ``None`` (e.g. a one-shot,
    exactly-sized cache with no padding), the ordinary static mask is used instead.

    ``mask_static_parts``: when given alongside ``context_valid_len_tt`` (one
    ``DynamicMaskStaticParts`` per distinct ``(is_causal, sliding_window)`` pair in
    ``layer_configs``, from ``attention.build_attention_mask_static_parts``), masks are
    recombined via ``combine_attention_mask_dynamic`` -- pure elementwise ops on
    already-built tensors, safe inside a captured Metal trace -- instead of being rebuilt
    from scratch (``ttnn.arange``/``ones``/``zeros``/``full``, each a host->device write)
    every call, which a trace capture rejects (``TT_FATAL: Writes are not supported during
    trace capture``). Required for ``generate.py``'s ``_traced_steady_state``; ordinary
    (non-traced) callers can omit it."""
    ctx_len = max_seq_len
    q_len = noise.shape[-2]
    # Must match dflash_attention_forward's own pad decision exactly (same flag) --
    # see _dflash_pad_noise_concat_enabled. When mask_static_parts is given, whoever
    # built it (generate.py, outside trace capture) is responsible for having already
    # passed the same q_len_padded to build_attention_mask_static_parts.
    q_len_padded = _tile_pad_len(q_len) if _dflash_pad_noise_concat_enabled() else None

    mask_cache: dict[tuple[bool, int | None], ttnn.Tensor] = {}

    def mask_for(is_causal, sliding_window):
        key = (is_causal, sliding_window)
        if key not in mask_cache:
            if mask_static_parts is not None:
                mask_cache[key] = combine_attention_mask_dynamic(mask_static_parts[key], context_valid_len_tt)
            elif context_valid_len_tt is not None:
                mask_cache[key] = build_attention_mask_additive_device_dynamic(
                    mesh_device, ctx_len, q_len, is_causal, sliding_window, context_valid_len_tt, q_len_padded
                )
            else:
                mask_cache[key] = build_attention_mask_additive_device(
                    mesh_device, ctx_len, q_len, is_causal, sliding_window, q_len_padded
                )
        return mask_cache[key]

    for (k_cache, v_cache), layer_weights, (is_causal, sliding_window) in zip(kv_caches, weights.layers, layer_configs):
        mask = mask_for(is_causal, sliding_window)
        noise = dflash_layer_forward(
            k_cache,
            v_cache,
            noise,
            layer_weights,
            cos_noise,
            sin_noise,
            mask,
            mesh_config,
            ccl_manager,
            num_local_heads,
            num_local_kv_heads,
            head_dim,
            eps,
        )
    return noise


def dflash_drafter_update_kv_caches(
    delta: ttnn.Tensor,  # [1,1,>=length,hidden] -- this iteration's context tap (context.py output)
    length: int,
    weights: Gemma4DFlashWeights,
    cos_delta: ttnn.Tensor,  # [1,1,>=length,head_dim]
    sin_delta: ttnn.Tensor,
    kv_caches: list[tuple[ttnn.Tensor, ttnn.Tensor]],
    offset: int,
    num_local_heads: int,
    num_local_kv_heads: int,
    head_dim: int,
    eps: float,
) -> None:
    """Project this iteration's newly-committed-token context tap through EVERY drafter
    layer's own k_proj/v_proj/k_norm (via each layer's fused wqkv weight) + RoPE, and
    write the result into that layer's persistent K/V cache at [offset:offset+length] --
    see attention.py's ``project_and_cache_context_delta`` (this just loops it over every
    layer, since each layer has its own independent attention weights and therefore its
    own independent cached K/V, even though they all start from the same ``delta``)."""
    for (k_cache, v_cache), layer_weights in zip(kv_caches, weights.layers):
        project_and_cache_context_delta(
            delta,
            length,
            layer_weights.attn,
            cos_delta,
            sin_delta,
            k_cache,
            v_cache,
            offset,
            num_local_heads,
            num_local_kv_heads,
            head_dim,
            eps,
        )

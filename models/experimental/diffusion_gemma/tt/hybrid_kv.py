# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DG-local model-owned hybrid KV cache setup.

The serving path still owns one cache for one active sequence, but the physical
layout is paged so the 25 sliding-attention layers retain only their 1024-token
window.  The five full-attention layers retain the complete served context.

This composes over Gemma4's existing bounded paged-cache implementation without
editing the shared backbone.  Page tables are deterministic identity mappings:
there is no vLLM block-pool ownership or concurrent-request claim here.
"""

from __future__ import annotations

import math

from loguru import logger

import ttnn
from models.demos.gemma4.tt.attention.kv_cache_hybrid import build_hybrid_page_tables
from models.tt_transformers.tt.common import PagedAttentionConfig


DEFAULT_BLOCK_SIZE = 64


def model_owned_hybrid_kv_model_kwargs(
    *,
    max_seq_len: int,
    max_batch_size: int = 1,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> dict:
    """Model-construction kwargs for one-sequence bounded hybrid KV."""

    max_seq_len = int(max_seq_len)
    max_batch_size = int(max_batch_size)
    block_size = int(block_size)
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
    if max_batch_size != 1:
        raise ValueError(
            "DG model-owned hybrid KV currently supports max_batch_size=1; "
            "concurrent sequences require vLLM block-pool ownership"
        )
    if block_size <= 0 or block_size % ttnn.TILE_SIZE != 0:
        raise ValueError(f"hybrid KV block_size must be a positive {ttnn.TILE_SIZE}-token multiple")

    blocks_per_sequence = math.ceil(max_seq_len / block_size)
    return {
        "create_kv_cache": True,
        "bounded_sliding_kv_cache": True,
        "paged_attention_config": PagedAttentionConfig(
            block_size=block_size,
            max_num_blocks=blocks_per_sequence * max_batch_size,
        ),
    }


def attach_model_owned_hybrid_kv(
    tt_model,
    *,
    max_seq_len: int,
    max_batch_size: int = 1,
    block_size: int = DEFAULT_BLOCK_SIZE,
):
    """Attach identity page tables and zero-copy full-layer sequence views.

    The model must already have been constructed with the kwargs returned by
    :func:`model_owned_hybrid_kv_model_kwargs`.
    """

    max_seq_len = int(max_seq_len)
    max_batch_size = int(max_batch_size)
    block_size = int(block_size)
    if max_batch_size != 1:
        raise ValueError("DG model-owned hybrid KV page tables currently support one active sequence")

    text_config = getattr(tt_model.hf_config, "text_config", tt_model.hf_config)
    layer_types = list(getattr(text_config, "layer_types", ()))[: len(tt_model.layers)]
    if len(layer_types) != len(tt_model.layers):
        raise ValueError(
            f"hybrid KV needs one layer type per model layer: {len(layer_types)} != {len(tt_model.layers)}"
        )
    sliding_window = int(getattr(text_config, "sliding_window", 0) or 0)
    if sliding_window <= 0 or sliding_window % block_size != 0:
        raise ValueError(f"sliding_window must be a positive multiple of block_size={block_size}, got {sliding_window}")
    if len(tt_model.tt_kv_cache) != len(tt_model.layers):
        raise ValueError(f"hybrid KV cache/layer mismatch: {len(tt_model.tt_kv_cache)} != {len(tt_model.layers)}")

    sliding_mask = [layer_type == "sliding_attention" for layer_type in layer_types]
    unsupported = [
        (index, layer_type)
        for index, layer_type in enumerate(layer_types)
        if layer_type not in ("sliding_attention", "full_attention")
    ]
    if unsupported:
        raise ValueError(f"unsupported hybrid KV layer types: {unsupported}")

    page_tables = build_hybrid_page_tables(
        len(layer_types),
        sliding_mask,
        num_users=max_batch_size,
        block_size=block_size,
        max_seq_len=max_seq_len,
        sliding_window=sliding_window,
    )

    max_blocks = math.ceil(max_seq_len / block_size)
    sliding_blocks = sliding_window // block_size
    full_views = {}
    logical_spans = []
    for layer_idx, (is_sliding, cache_pair) in enumerate(zip(sliding_mask, tt_model.tt_kv_cache)):
        expected_blocks = sliding_blocks if is_sliding else max_blocks
        logical_spans.append(sliding_window if is_sliding else max_seq_len)
        for kind, cache in zip(("K", "V"), cache_pair):
            if int(cache.shape[0]) != expected_blocks:
                raise ValueError(
                    f"layer {layer_idx} {kind} cache has {cache.shape[0]} blocks, expected {expected_blocks}; "
                    "construct the model with model_owned_hybrid_kv_model_kwargs"
                )
            if int(cache.shape[2]) != block_size:
                raise ValueError(
                    f"layer {layer_idx} {kind} cache block axis is {cache.shape[2]}, expected {block_size}"
                )
        if not is_sliding:
            k_cache, v_cache = cache_pair
            if int(k_cache.shape[1]) != 1 or int(v_cache.shape[1]) != 1:
                raise ValueError(
                    f"full-attention layer {layer_idx} needs one local KV head for zero-copy flattening, "
                    f"got K={k_cache.shape[1]} V={v_cache.shape[1]}"
                )
            allocated_span = int(k_cache.shape[0]) * int(k_cache.shape[2])
            full_views[layer_idx] = (
                ttnn.reshape(k_cache, (1, 1, allocated_span, int(k_cache.shape[3]))),
                ttnn.reshape(v_cache, (1, 1, allocated_span, int(v_cache.shape[3]))),
            )

    to_device = getattr(tt_model, "_page_tables_to_ttnn", None)
    device_page_tables = to_device(page_tables) if callable(to_device) else page_tables

    tt_model._dg_model_owned_hybrid_kv = True
    tt_model._dg_hybrid_host_page_tables_per_layer = page_tables
    tt_model._dg_hybrid_page_tables_per_layer = device_page_tables
    tt_model._dg_hybrid_sliding_layers = frozenset(i for i, value in enumerate(sliding_mask) if value)
    tt_model._dg_hybrid_full_cache_views = full_views
    tt_model._dg_hybrid_logical_spans = tuple(logical_spans)
    tt_model._dg_hybrid_block_size = block_size
    tt_model._dg_hybrid_max_seq_len = max_seq_len
    tt_model._dg_hybrid_sliding_window = sliding_window

    full_count = len(layer_types) - sum(sliding_mask)
    logger.info(
        f"[DiffusionGemma hybrid KV] attached model-owned paged cache: "
        f"sliding={sum(sliding_mask)}x{sliding_window}, full={full_count}x{max_seq_len}, "
        f"block_size={block_size}"
    )
    return device_page_tables


def page_tables_per_layer(tt_model):
    """Return the attached model-owned hybrid page tables, or ``None``."""

    return getattr(tt_model, "_dg_hybrid_page_tables_per_layer", None)

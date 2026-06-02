# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-style hybrid kv_cache_groups page-table construction for Gemma4.

vLLM's hybrid KV-cache manager splits attention layers into groups by their
KvCacheSpec. For Gemma4 the relevant split is:

  - FullAttentionSpec layers  → allocate max_model_len/block_size blocks per
    sequence; per-layer page_table is the full block-id sequence.
  - SlidingWindowSpec layers  → allocate only sliding_window/block_size blocks
    per sequence, but the per-layer page_table is still
    max_model_len/block_size wide. The first sliding_window/block_size entries
    are valid block IDs; the tail is zero-padded.

The kernels treat position 0 as "block 0", so the zero-padded tail would
silently clobber block 0 once decode positions cross the sliding window. The
cache_position_modulo kwarg on paged_update_cache /
paged_scaled_dot_product_attention_decode / paged_fill_cache makes the
sliding-spec layers wrap their position into [0, sliding_window) before the
page_table lookup, so absolute positions past the boundary land back inside
the valid prefix instead of on the padded tail. This file constructs that
page_table.

The helper is layer-type-aware but layer-index-agnostic — the caller passes
a sliding_layers_mask of length num_layers. Gemma4 uses the 4-sliding-then-
1-full pattern (via hf_config.layer_types); no assumption about that pattern
is made here. Kv-shared layers (Gemma4-E2B / -E4B) are also a caller concern:
init_hybrid_kv_caches still allocates one cache per layer, and the bridge's
kv-share alias logic re-points sink layers' buffer + page_table after the
fact.
"""

from __future__ import annotations

import math

import torch

import ttnn

from .kv_cache import init_kv_cache


def _max_blocks_per_seq(max_seq_len: int, block_size: int) -> int:
    return math.ceil(max_seq_len / block_size)


def build_hybrid_page_tables(
    num_layers: int,
    sliding_layers_mask: list[bool],
    *,
    num_users: int,
    block_size: int,
    max_seq_len: int,
    sliding_window: int | None,
) -> list[torch.Tensor]:
    """Build a per-layer page_table list mirroring vLLM's hybrid kv_cache_groups.

    For every layer the returned page_table has shape ``[num_users, max_blocks_per_req]``
    where ``max_blocks_per_req = ceil(max_seq_len / block_size)``. Sliding-window
    layers (mask[i]=True) only use the first ``sliding_window/block_size`` entries
    of each row; the rest is zero-padded. Full-attention layers use the full row.

    Block IDs are allocated independently per layer (each layer has its own
    physical pool). For full layers IDs run ``[0, max_blocks_per_req * num_users)``;
    for sliding layers IDs run ``[0, sliding_blocks * num_users)``.

    Returns a list of CPU int32 torch tensors. Caller is responsible for
    moving them to the device.
    """
    assert len(sliding_layers_mask) == num_layers, "sliding_layers_mask must have one entry per layer"

    max_blocks_per_req = _max_blocks_per_seq(max_seq_len, block_size)
    if sliding_window is not None:
        if sliding_window % block_size != 0:
            raise ValueError(
                f"sliding_window ({sliding_window}) must be a multiple of block_size ({block_size}) "
                "for the bounded paged path"
            )
        sliding_blocks = sliding_window // block_size
    else:
        sliding_blocks = max_blocks_per_req

    page_tables: list[torch.Tensor] = []
    for i in range(num_layers):
        pt = torch.zeros(num_users, max_blocks_per_req, dtype=torch.int32)
        if sliding_layers_mask[i]:
            # SlidingWindowSpec: only the first sliding_blocks entries are valid; the
            # rest is the zero-padded tail that cache_position_modulo wraps over.
            valid = sliding_blocks
            for u in range(num_users):
                pt[u, :valid] = torch.arange(u * valid, (u + 1) * valid, dtype=torch.int32)
        else:
            # FullAttentionSpec: every entry is a valid block.
            for u in range(num_users):
                pt[u, :] = torch.arange(u * max_blocks_per_req, (u + 1) * max_blocks_per_req, dtype=torch.int32)
        page_tables.append(pt)
    return page_tables


def init_hybrid_kv_caches(
    mesh_device,
    attention_configs,
    sliding_layers_mask: list[bool],
    paged_attention_config,
    *,
    max_batch_size: int = 1,
    max_seq_len: int = 131072,
    cache_dtype=ttnn.bfloat16,
    tensor_cache_path=None,
):
    """Allocate per-layer KV caches with vLLM-style hybrid sizing.

    Full layers reuse ``paged_attention_config.max_num_blocks``. Sliding layers
    are sized to ``sliding_window/block_size * max_batch_size`` — the bounded
    SlidingWindowSpec allocation. Returns a list of ``[k_cache, v_cache]`` in
    layer order, ready to alias / hand to the model as the per-layer kv_cache.

    Kv-shared layers (sinks) still get their own cache here; the caller is
    expected to alias sink-layer caches to their source via the model's
    ``kv_shared_layer_map`` after this returns. Matches Gemma4ForCausalLM's
    ``allocate_kv_cache_per_layer`` flow.
    """
    assert len(attention_configs) == len(
        sliding_layers_mask
    ), "attention_configs and sliding_layers_mask must have the same length"

    block_size = paged_attention_config.block_size
    kv_caches = []
    for cfg, is_sliding in zip(attention_configs, sliding_layers_mask):
        if is_sliding and cfg.sliding_window is not None:
            sliding_blocks_per_seq = cfg.sliding_window // block_size
            max_num_blocks_override = sliding_blocks_per_seq * max_batch_size
        else:
            max_num_blocks_override = None
        kv_caches.append(
            init_kv_cache(
                mesh_device=mesh_device,
                config=cfg,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                paged_attention_config=paged_attention_config,
                cache_dtype=cache_dtype,
                tensor_cache_path=tensor_cache_path,
                max_num_blocks_override=max_num_blocks_override,
            )
        )
    return kv_caches


__all__ = ["build_hybrid_page_tables", "init_hybrid_kv_caches"]

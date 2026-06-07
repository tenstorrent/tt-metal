# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Paged KV cache for the TP4 dots.ocr prefill/decode.

Reuses the proven ``TTNNPagedAttentionKVCache`` from tt_symbiote. The TP4 twist:
each chip stores exactly its ONE assigned KV head (head 0 on chips 0-1, head 1
on chips 2-3), so the cache is created with ``num_kv_heads=1`` and allocated
per-chip via ``ttnn.zeros`` on the mesh (each device gets its own buffer). Every
chip fills/reads its own shard independently; chips that share a KV head compute
identical K/V from the replicated hidden, so their caches agree.
"""

import torch

from models.experimental.tt_symbiote.modules.attention import (
    PagedAttentionConfig,
    TTNNPagedAttentionKVCache,
)


def create_paged_kv_cache(config, mesh_device, batch_size: int = 1, block_size: int = 64, max_num_blocks: int = 2048):
    """Build + place a paged KV cache sized for the TP4 per-chip layout."""
    paged_cfg = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks, batch_size=batch_size)
    cache = TTNNPagedAttentionKVCache(
        num_layers=config.num_hidden_layers,
        num_kv_heads=1,  # per chip: this chip's single assigned KV head
        head_dim=config.head_dim,
        config=paged_cfg,
        device=mesh_device,
        dtype=torch.bfloat16,
    )
    cache.to_device(mesh_device)
    return cache

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only diagnostic pool geometry shared by batch and prefill probes."""
import torch


def shared_pool_layout(batch_size, capacity, attention_layers):
    if not 1 <= batch_size <= 32 or capacity < 35 or attention_layers < 1:
        raise ValueError("Expected batch1..32, context at least35 and indexed attention")
    pages_per_slot = ((capacity + 127) // 128) * 4
    blocks = batch_size * pages_per_slot
    assert blocks >= 2 * batch_size, "Pool must cover all temporary warmup reservations"
    specs = [((blocks, 1, 32, 256), torch.bfloat16, index) for index in range(attention_layers)]
    base = torch.arange(blocks, dtype=torch.int32).reshape(batch_size, pages_per_slot)
    tables = [(base + layer * pages_per_slot) % blocks for layer in range(attention_layers)]
    return specs, tables


def indexed_attention_layout(cache, batch_size, capacity):
    """Enumerate actual cache layers in decoder order, including reduced models."""
    attention = [layer.attention for layer in cache.layers if layer.attention is not None]
    return shared_pool_layout(batch_size, capacity, len(attention))


def validate_pool_aliases(caches, pool, specs):
    """Every indexed layer, physical slot and scratch must alias its own pair."""
    assert len(pool.pairs) == len(specs)
    pairs = [pool.pair_for_attention_layer(index) for index in range(len(specs))]
    assert len({id(tensor) for pair in pairs for tensor in pair}) == 2 * len(specs)
    for cache in caches:
        if cache is None:
            continue
        assert cache.shared_kv and cache.kv_bound
        attention = [layer.attention for layer in cache.layers if layer.attention is not None]
        assert len(attention) == len(specs)
        for index, state in enumerate(attention):
            assert state.k_cache is pairs[index][0] and state.v_cache is pairs[index][1]

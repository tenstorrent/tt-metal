# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Native two-call batched K/V prefill fill; no model hooks or device import.

Caller prepares batch_indices once and establishes non-overlapping destination
pages for the active writes. Arbitrary/aliased external tables retain the old
ordered loop unless that ownership guarantee is independently established.
"""

import math


def batched_fill_pair(
    key_cache, value_cache, key, value, page_table, batch_indices, *, start_pos=0, disjoint_write_pages=False
):
    """Fill already-padded/cast K/V using native raw tile copies, in place.

    Only aligned bulk chunks enter here. The existing public scheduler keeps
    nonaligned starts on its token-wise update path; no partial page is clobbered.
    No dtype conversion, cache-layout change, or index allocation occurs here.
    """
    import ttnn

    assert disjoint_write_pages, "Caller must establish disjoint active physical cache pages"
    assert start_pos >= 0 and start_pos % 32 == 0
    batch, heads, seq, width = tuple(key.shape)
    assert 1 <= batch <= 32 and heads == 1 and width == 256 and seq % 32 == 0
    assert tuple(value.shape) == tuple(key.shape)
    assert tuple(key_cache.shape) == tuple(value_cache.shape)
    assert tuple(key_cache.shape)[1:] == (1, 32, 256)
    assert key_cache.buffer_address() != value_cache.buffer_address()
    assert key_cache.dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b)
    for tensor in (key_cache, value_cache, key, value):
        assert tensor.dtype == key_cache.dtype and tensor.layout == ttnn.TILE_LAYOUT
        assert tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        assert tensor.device() == key_cache.device()
    assert all(c.buffer_address() != t.buffer_address() for c in (key_cache, value_cache) for t in (key, value))
    assert len(page_table.shape) == 2 and page_table.shape[0] >= batch
    assert page_table.shape[1] >= (start_pos + seq) // 32
    assert page_table.dtype == ttnn.int32 and page_table.layout == ttnn.ROW_MAJOR_LAYOUT
    assert batch_indices.dtype in (ttnn.int32, ttnn.uint32) and batch_indices.layout == ttnn.ROW_MAJOR_LAYOUT
    assert math.prod(batch_indices.padded_shape) == batch
    for tensor in (page_table, batch_indices):
        assert tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG and tensor.device() == key_cache.device()
    table = page_table[:, start_pos // 32 :]
    ttnn.experimental.paged_fill_cache(key_cache, key, table, batch_idx_tensor=batch_indices)
    ttnn.experimental.paged_fill_cache(value_cache, value, table, batch_idx_tensor=batch_indices)
    return key_cache, value_cache

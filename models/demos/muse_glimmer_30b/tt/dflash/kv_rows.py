# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Write K/V rows of a decode-heads tensor into a paged cache with PagedUpdateRows (batch 1, P rows)."""

import ttnn
from models.demos.gemma4.custom_ops.paged_update_rows import PagedUpdateRows


def write_kv_rows(k_cache, v_cache, k_heads, v_heads, write_rows, page_table, *, num_kv_heads, block_size, batch=1):
    """k_heads, v_heads: [1, 32, num_kv_heads, hd] decode heads layout (any memory config). Row i is written at the
    cache position in write_rows[i] (device tensors [1] int32; -1 skips). Does not free k_heads / v_heads.
    batch=U > 1 (multi-user): write_rows = R tensors [U] (entry u = user u's position for its row p), page_table
    [U, nb] (row u = user u's page row), source row u * R + p = user u's row p."""
    for cache, src in ((k_cache, k_heads), (v_cache, v_heads)):
        # heads layout [1, 32 rows, kv_local, hd] -> [1, kv_local, 32 rows, hd] TILE bf16 interleaved
        # to_memory_config returns an alias when src already has the target config: skip it and never free src.
        s = (
            src
            if src.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            else ttnn.to_memory_config(src, ttnn.DRAM_MEMORY_CONFIG)
        )
        s2 = ttnn.to_layout(s, ttnn.ROW_MAJOR_LAYOUT) if s.layout != ttnn.ROW_MAJOR_LAYOUT else s
        p = ttnn.permute(s2, (0, 2, 1, 3))
        p2 = ttnn.to_layout(p, ttnn.TILE_LAYOUT)
        PagedUpdateRows.op(
            cache,
            p2,
            write_rows,
            page_table,
            batch=batch,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
        )
        for t in {id(t): t for t in (s, s2, p, p2)}.values():
            if t is not src:
                t.deallocate(True)

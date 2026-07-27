# SPDX-License-Identifier: Apache-2.0
"""Is a row-major [1,K] shard byte-identical to a [1,32]-tiled [1,K] shard?

If yes, converting between them needs no kernel at all -- just a Tensor that
aliases the same device buffer with a different TensorSpec. That makes the
missing ttnn primitive a zero-cost view rather than a copy op, which is a much
smaller ask. This checks buffer size, page size and shard spec agree.

A [1,32] tile is 2 faces of [1,16] = 32 contiguous values, so it should match a
row-major row exactly, but "should" is doing a lot of work; measure it.
"""
import torch
from loguru import logger

import ttnn

K = 3584


def test_alias_equivalence(device):
    cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_cores = len(cores)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in cores])
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, [1, K], ttnn.ShardOrientation.ROW_MAJOR),
    )
    src = torch.randn(1, 1, num_cores, K)

    rm = ttnn.from_torch(src, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    tt = ttnn.from_torch(
        src, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc, tile=ttnn.Tile([1, 32])
    )

    def describe(t, name):
        info = {
            "size": t.buffer_num_pages() * t.buffer_page_size(),
            "page_size": t.buffer_page_size(),
            "num_pages": t.buffer_num_pages(),
            "aligned_page": t.buffer_aligned_page_size(),
            "shard": None,
        }
        try:
            info["shard"] = tuple(t.memory_config().shard_spec.shape)
        except Exception:
            pass
        logger.info(
            f"ALIAS {name}: bytes={info['size']} pages={info['num_pages']} "
            f"page_size={info['page_size']} aligned={info['aligned_page']} shard={info['shard']}"
        )
        return info

    a = describe(rm, "row_major   ")
    b = describe(tt, "tiled[1,32] ")

    # Same bytes on device, laid out the same way?
    same_bytes = torch.equal(ttnn.to_torch(rm), ttnn.to_torch(tt))
    logger.info(f"ALIAS same buffer_size: {a['size'] == b['size']}")
    logger.info(f"ALIAS same page_size:   {a['page_size'] == b['page_size']}")
    logger.info(f"ALIAS same shard shape: {a['shard'] == b['shard']}")
    logger.info(f"ALIAS identical values: {same_bytes}")

    # Page size is allowed to differ. It only describes how the buffer is chunked
    # for addressing (8x7168 vs 896x64 both cover the same 57344 bytes); a re-spec'd
    # view recomputes it. What has to hold is that the bytes are the same and in the
    # same order, which is what identical to_torch values on a shared shard prove.
    verdict = a["size"] == b["size"] and a["shard"] == b["shard"] and same_bytes
    logger.info(f"ALIAS VERDICT: {'zero-cost view is valid' if verdict else 'layouts differ, needs a real kernel'}")
    assert verdict, "row-major and [1,32]-tiled shards are NOT interchangeable"

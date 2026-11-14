# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Expert weights in "column pages" (Laguna).

A stacked expert weight [1, E, K, N] stored as interleaved TILE pages puts the K tiles of one output column on
different DRAM banks, one 576-byte bfp4 tile per read; a core streaming a column issues K small reads (~260 GB/s for
64 cores). Here the same TILE tensor is DRAM ND-sharded with shard shape [1, 1, K, 32]: each (expert, tile column)
is one shard, its K tiles back to back in one bank (shards round-robin over the banks). Read as whole shards
("column pages", K * 576 bytes) a core gets ~360 GB/s.

It stays an ordinary TILE tensor: tile (e, k, c) is still page (e * Kt + k) * Ct + c of a TensorAccessor, so stock
ops (sparse_matmul, the prefill routed-expert op) read it unchanged; the column kernels take the address of the
column's first tile, page e * Kt * Ct + c, and read K * 576 bytes from there.
"""

from dataclasses import dataclass

import ttnn

TILE = 32


@dataclass
class ColumnPages:
    buf: ttnn.Tensor  # [1, E, K, N] bfp4 TILE, DRAM ND-sharded [1, 1, K, 32]
    E: int
    Kt: int  # tiles per column (the K dimension)
    Ct: int  # tile columns per expert (the N dimension)


def column_pages_memory_config(device, K):
    banks = device.dram_grid_size().x
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    return ttnn.MemoryConfig(ttnn.BufferType.DRAM, ttnn.NdShardSpec(ttnn.Shape([1, 1, K, TILE]), grid))


def to_column_pages(w):
    """w: [1, E, K, N] bfp4 TILE DRAM -> the same values ND-sharded by column (a new tensor; w is left as is)."""
    assert w.dtype == ttnn.bfloat4_b and w.layout == ttnn.TILE_LAYOUT, (w.dtype, w.layout)
    K = w.padded_shape[-2]
    return ttnn.to_memory_config(w, column_pages_memory_config(w.device(), K))


def column_pages(w):
    """View an already column-sharded weight as ColumnPages."""
    return ColumnPages(w, w.shape[1], w.padded_shape[-2] // TILE, w.padded_shape[-1] // TILE)

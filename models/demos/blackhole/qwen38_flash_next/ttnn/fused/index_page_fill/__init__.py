# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``index_page_fill``: prefill's compressed index rows into the paged index pool of one lane.

``src`` ([1, 1, n * 32, 128] bf16 TILE, n >= 1 tile-rows of 32 compressed rows) goes into ``pool`` ([NB, 1, block_rows,
128] bf16 TILE, block_rows = block_size / 4) at logical tile-row L = (P >> 7) + i for source tile-row i, with P the
token position read from the device scalar ``position`` (uint32 [1, 1, 1, 1] ROW_MAJOR, a multiple of 128).  Logical
tile-row L sits at physical tile-row ``page_table[lane][L >> log2(tpb)] * tpb + (L & (tpb - 1))``, tpb = block_rows /
32.  Whole tiles only, so a chunk may start in the middle of a block.  P is read on device, so one compiled program
serves every start position of a captured trace.  Not registered (no composed twin in the decode chain).
"""

from __future__ import annotations

import ttnn

from .. import program as fp
from ..qsa_block import page_table_blocks

NAME = "index_page_fill"
FILL = fp.kernel_source(NAME, "fill_tiles.cpp")
FILL_CORES = 8  # cores 0..7 of core row 0; core j takes the tile-rows j, j + 8, ...
WIDTH = 128
ROW_TILES = WIDTH // fp.TILE
TILE_BYTES = fp.TILE * fp.TILE * 2
ALIGN_SLACK = 64


def _check(src, pool, page_table, position, lane: int, block_rows: int) -> tuple[int, int, int]:
    """(tile_rows, tpb_shift, max_blocks)."""

    if block_rows < fp.TILE or block_rows % fp.TILE or block_rows & (block_rows - 1):
        raise ValueError(f"block_rows must be a power of two and a multiple of {fp.TILE}, got {block_rows}")
    for label, tensor in (("src", src), ("pool", pool)):
        if tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"{label} must be TILE bf16, got {tensor.dtype} {tensor.layout}")
    src_shape, pool_shape = tuple(src.shape), tuple(pool.shape)
    if len(src_shape) != 4 or src_shape[:2] != (1, 1) or src_shape[3] != WIDTH or src_shape[2] % fp.TILE:
        raise ValueError(f"src must be [1, 1, n * {fp.TILE}, {WIDTH}], got {src_shape}")
    if src_shape[2] < fp.TILE:
        raise ValueError(f"src needs at least one tile-row, got {src_shape}")
    if len(pool_shape) != 4 or pool_shape[1:] != (1, block_rows, WIDTH):
        raise ValueError(f"pool must be [NB, 1, {block_rows}, {WIDTH}], got {pool_shape}")
    if lane < 0:
        raise ValueError(f"lane must be non-negative, got {lane}")
    if (
        tuple(position.shape) != (1, 1, 1, 1)
        or position.dtype != ttnn.uint32
        or position.layout != ttnn.ROW_MAJOR_LAYOUT
    ):
        raise ValueError(
            f"position must be a ROW_MAJOR uint32 [1, 1, 1, 1], got {position.dtype} {position.layout} {tuple(position.shape)}"
        )
    max_blocks = page_table_blocks(page_table, block_rows * 4, lane + 1)
    tile_rows = src_shape[2] // fp.TILE
    return tile_rows, (block_rows // fp.TILE).bit_length() - 1, max_blocks


def index_page_fill(src, pool, page_table, position, *, lane: int = 0, block_rows: int):
    """Write ``src``'s tile-rows into ``pool`` at lane ``lane``'s pages of ``page_table`` (int32 ROW_MAJOR
    [1, 1, >= lane + 1, max_blocks]) from the logical tile-row ``position >> 7``; returns ``pool`` (updated in place).
    """

    tile_rows, tpb_shift, max_blocks = _check(src, pool, page_table, position, lane, block_rows)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(FILL_CORES - 1, 0))])
    tensors = (src, pool, page_table, position)
    addresses = [t.buffer_address() for t in tensors]
    kernel = fp.reader_kernel(
        FILL,
        cores,
        [a for t in tensors for a in fp.accessor_args(t)],
        [
            (ttnn.CoreCoord(j, 0), [*addresses, tile_rows, lane, tpb_shift, max_blocks, j, FILL_CORES])
            for j in range(FILL_CORES)
        ],
    )
    buffer_bytes = ROW_TILES * TILE_BYTES + ((max_blocks * 4 + 63) & ~63) + 64 + ALIGN_SLACK
    cbs = [fp.cb_descriptor(0, ttnn.uint32, buffer_bytes, 1, cores)]
    meta = fp.program_meta(
        NAME,
        "fill",
        1,
        reads=(page_table, position),
        partial=((src, tile_rows * ROW_TILES * TILE_BYTES), (pool, tile_rows * ROW_TILES * TILE_BYTES)),
        cores=FILL_CORES,
    )
    return fp.run_program([src, page_table, position, pool], fp.program_descriptor([kernel], cbs=cbs), meta=meta)

# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``kv_page_scatter``: the B=1 generic packed KV cache's first rows into the paged KV pool of one lane.

Copies rows [0, P) of ``src`` ([1, 1, C, 512] ROW_MAJOR bf16) into ``pool`` ([1, 1, num_pages * bs, 512]) at row t ->
``page_table[lane][t >> shift] * bs + (t & (bs - 1))``, in whole 32-row groups (P rounded up to 32).  Data movement
only (the prompt's KV handed to the plain lane body's pool).  P, the lane, the shift and bs are runtime arguments, so
one compiled program serves every prompt of a page size.  ``src_row0`` (runtime argument, default 0 = the B=1 hand-off) shifts the
source window to rows [src_row0, src_row0 + rows): a multi-user pack's user ``u`` sits at row ``u * S`` of the dense cache
(``src_row0`` a multiple of 32).  Not registered (no composed twin in the decode chain);
``kv_page_scatter_composed`` is the reference.

``position`` (optional UINT32 [1, 1, 1, 1] ROW_MAJOR device scalar P, a multiple of 32) moves the destination: group g
lands at logical rows [P + 32 g, P + 32 g + 32) of the lane, P read on device, so a captured trace writes each replay's
chunk where the scalar says.  ``lane``, ``rows`` and ``src_row0`` stay host values.  The page-table range check covers
[0, rows) only; the caller keeps P + rows inside the lane's entries.
"""

from __future__ import annotations

import ttnn

from .. import program as fp
from ..qsa_block import page_shift, page_table_blocks

NAME = "kv_page_scatter"
SCATTER = fp.kernel_source(NAME, "scatter_rows.cpp")
SCATTER_CORES = 8  # cores 0..7 of core row 0; core j takes the 32-row groups j, j + 8, ...
KV_WIDTH = 512
GROUP_ROWS = fp.TILE
ROW_BYTES = KV_WIDTH * 2
GROUP_BYTES = GROUP_ROWS * ROW_BYTES
PT_ALIGN_SLACK = 64
POSITION_SLOT_BYTES = 128  # the aligned slot of P after the page table row, plus the alignment of the slot's start


def _check(src, pool, page_table, lane: int, rows: int, page_size: int, src_row0: int = 0) -> tuple[int, int]:
    for label, tensor in (("src", src), ("pool", pool)):
        shape = tuple(tensor.shape)
        if (
            len(shape) != 4
            or shape[:2] != (1, 1)
            or shape[3] != KV_WIDTH
            or tensor.dtype != ttnn.bfloat16
            or tensor.layout != ttnn.ROW_MAJOR_LAYOUT
        ):
            raise ValueError(f"{label} must be ROW_MAJOR bf16 [1, 1, rows, {KV_WIDTH}], got {tensor.layout} {shape}")
    max_blocks = page_table_blocks(page_table, page_size, lane + 1)
    groups = -(-rows // GROUP_ROWS)
    if src_row0 < 0 or src_row0 % GROUP_ROWS:
        raise ValueError(f"src_row0 must be a non-negative multiple of {GROUP_ROWS}, got {src_row0}")
    if rows < 1 or src_row0 + groups * GROUP_ROWS > tuple(src.shape)[2]:
        raise ValueError(f"src_row0 {src_row0} + rows {rows} (rounded up to 32) must be in [1, {tuple(src.shape)[2]}]")
    if tuple(pool.shape)[2] % page_size:
        raise ValueError(f"the pool's {tuple(pool.shape)[2]} rows must be a multiple of page_size {page_size}")
    if ((groups * GROUP_ROWS - 1) >> page_shift(page_size)) >= max_blocks:
        raise ValueError(f"rows {rows} need more pages than the page table's {max_blocks} entries")
    return groups, max_blocks


def _check_position(position) -> None:
    if (
        tuple(position.shape) != (1, 1, 1, 1)
        or position.dtype != ttnn.uint32
        or position.layout != ttnn.ROW_MAJOR_LAYOUT
    ):
        raise ValueError(
            f"position must be a UINT32 ROW_MAJOR [1, 1, 1, 1] scalar, got {position.dtype} {position.layout} "
            f"{tuple(position.shape)}"
        )


def kv_page_scatter(src, pool, page_table, *, lane: int, rows: int, page_size: int, src_row0: int = 0, position=None):
    """Scatter ``src`` rows [src_row0, src_row0 + rows) (up to 32-row groups) into ``pool`` at lane ``lane``'s pages of ``page_table``
    (int32 ROW_MAJOR [1, 1, >= lane + 1, max_blocks]); returns ``pool`` (updated in place).  ``position`` (device scalar,
    see the module doc) offsets the destination rows by P read on device."""

    groups, max_blocks = _check(src, pool, page_table, lane, rows, page_size, src_row0)
    if position is not None:
        _check_position(position)
    shift = page_shift(page_size)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(SCATTER_CORES - 1, 0))])
    tensors = (src, pool, page_table)
    addresses = [t.buffer_address() for t in tensors]
    pos_args = [] if position is None else [position.buffer_address()]
    pos_tensors = [] if position is None else [position]
    kernel = fp.reader_kernel(
        SCATTER,
        cores,
        [a for t in (*tensors, *pos_tensors) for a in fp.accessor_args(t)],
        [
            (
                ttnn.CoreCoord(j, 0),
                list([*addresses, groups, lane, shift, page_size, max_blocks, j, SCATTER_CORES, src_row0]) + pos_args,
            )
            for j in range(SCATTER_CORES)
        ],
        defines=[] if position is None else [("DEVICE_POSITION", "1")],
    )
    cb_bytes = GROUP_BYTES + max_blocks * 4 + PT_ALIGN_SLACK + (0 if position is None else POSITION_SLOT_BYTES)
    cbs = [fp.cb_descriptor(0, ttnn.uint32, cb_bytes, 1, cores)]
    meta = fp.program_meta(
        NAME,
        "scatter_rows",
        1,
        reads=(page_table, *pos_tensors),
        partial=((src, groups * GROUP_BYTES), (pool, groups * GROUP_BYTES)),
        cores=SCATTER_CORES,
    )
    return fp.run_program([src, page_table, *pos_tensors, pool], fp.program_descriptor([kernel], cbs=cbs), meta=meta)


def kv_page_scatter_composed(
    src, pool, page_table, *, lane: int, rows: int, page_size: int, page_row=None, src_row0: int = 0
):
    """The reference: per 32-row group, ``ttnn.slice`` of the source rows and ``update_padded_kv_cache`` of the slab at
    the group's physical row (the lane body's own cache write).  ``page_row`` = lane's page-table row as host ints
    (read from replica 0 of ``page_table`` when omitted)."""

    import torch

    groups, max_blocks = _check(src, pool, page_table, lane, rows, page_size, src_row0)
    if page_row is None:
        row = ttnn.to_torch(ttnn.get_device_tensors(page_table)[0])
        page_row = [int(v) for v in row[0, 0, lane, :max_blocks]]
    mesh = src.device()
    shift = page_shift(page_size)
    zero = ttnn.from_torch(
        torch.zeros(1, 1, 1, 1, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    for g in range(groups):
        t0 = g * GROUP_ROWS
        start = page_row[t0 >> shift] * page_size + (t0 & (page_size - 1))
        slab = ttnn.slice(
            src,
            (0, 0, src_row0 + t0, 0),
            (1, 1, src_row0 + t0 + GROUP_ROWS, KV_WIDTH),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        row_start = ttnn.from_torch(
            torch.full((1, 1, 1, 1), start, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(pool, slab, zero, row_start, 0, 1, 0)
        ttnn.deallocate(slab)
        ttnn.deallocate(row_start)
    ttnn.deallocate(zero)
    return pool

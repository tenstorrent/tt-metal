# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``pool_blocks``: whole-block moves of the paged QSA pools (one layer slot, one device).

K/V pool ``[1, 1, NB * bs, 512]`` BF16 ROW_MAJOR (block b = rows [b * bs, (b + 1) * bs), one 1024-byte row per page) and
index pool ``[NB, 1, bs / 4, 128]`` BF16 TILE (block b = the tile ids [b * bs / 32, (b + 1) * bs / 32)), both DRAM
interleaved, bs a power of two in [128, 4096].

* ``copy_blocks`` / ``zero_blocks``: block ``src`` -> block ``dst`` in both pools, or zero-fill of ``dst``
  (``src == ZERO_SOURCE``); ``src == dst`` is skipped.  The dst blocks must be distinct and none may be the src of
  another pair, so one program serves the whole list (up to ``MAX_PAIRS`` pairs per program, longer lists loop).
* ``state_to_block`` / ``block_to_state``: the raw bytes of a DRAM-interleaved tensor (pages in page-id order, or only
  batch row ``batch_index`` of a ``[B, ...]`` tensor) to or from the rows of one K/V pool block from ``byte_offset``
  (a multiple of 1024).  The state page size P must be a multiple of 1024 bytes, or divide 1024 and be a multiple of
  64.  Only the state's bytes are written; the rest of the last 1024-byte pool row is untouched.
Data movement only; eager use (host block ids are runtime arguments).  Not registered.
"""

from __future__ import annotations

import math

import ttnn

from .. import program as fp

NAME = "pool_blocks"
COPY = fp.kernel_source(NAME, "copy_blocks.cpp")
STATE = fp.kernel_source(NAME, "state_copy.cpp")
CORES = 8  # cores 0..7 of core row 0
ZERO_SOURCE = 0xFFFFFFFF
MAX_PAIRS = 48  # pairs per program (runtime arguments, padded to this many)
ROW_BYTES = 1024
GROUP_ROWS = fp.TILE
KV_WIDTH = 512
INDEX_WIDTH = 128
TILE_BYTES = 2048
UNIT_BYTES = GROUP_ROWS * ROW_BYTES + TILE_BYTES
ALIGN_SLACK = 64


def _cores() -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(CORES - 1, 0))])


def _check_block_size(block_size: int) -> int:
    if block_size < 128 or block_size > 4096 or block_size & (block_size - 1):
        raise ValueError(f"block_size must be a power of two in [128, 4096], got {block_size}")
    return block_size.bit_length() - 1


def _in_dram_interleaved(tensor) -> bool:
    config = tensor.memory_config()
    return config.buffer_type == ttnn.BufferType.DRAM and config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


def _check_pools(kv_pool, index_pool, block_size: int) -> int:
    """The block count NB."""

    _check_block_size(block_size)
    kv, idx = tuple(kv_pool.shape), tuple(index_pool.shape)
    if (
        len(kv) != 4
        or kv[:2] != (1, 1)
        or kv[3] != KV_WIDTH
        or kv_pool.dtype != ttnn.bfloat16
        or kv_pool.layout != ttnn.ROW_MAJOR_LAYOUT
        or kv[2] % block_size
    ):
        raise ValueError(
            f"kv_pool must be ROW_MAJOR bf16 [1, 1, NB * {block_size}, {KV_WIDTH}], got {kv_pool.layout} {kv}"
        )
    blocks = kv[2] // block_size
    if (
        idx != (blocks, 1, block_size // 4, INDEX_WIDTH)
        or index_pool.dtype != ttnn.bfloat16
        or index_pool.layout != ttnn.TILE_LAYOUT
    ):
        raise ValueError(
            f"index_pool must be TILE bf16 [{blocks}, 1, {block_size // 4}, {INDEX_WIDTH}], got {index_pool.layout} {idx}"
        )
    if not (_in_dram_interleaved(kv_pool) and _in_dram_interleaved(index_pool)):
        raise ValueError("the pools must be DRAM interleaved")
    return blocks


def _check_pairs(pairs, blocks: int) -> list[tuple[int, int]]:
    """The pairs that move data (src != dst), validated."""

    live = []
    for src, dst in pairs:
        src, dst = int(src), int(dst)
        if not 0 <= dst < blocks or not (src == ZERO_SOURCE or 0 <= src < blocks):
            raise ValueError(f"pair ({src}, {dst}) is outside the {blocks} blocks")
        if src != dst:
            live.append((src, dst))
    dsts = [dst for _, dst in live]
    if len(set(dsts)) != len(dsts):
        raise ValueError(f"the dst blocks must be distinct, got {dsts}")
    clash = set(dsts) & {src for src, _ in live if src != ZERO_SOURCE}
    if clash:
        raise ValueError(f"blocks {sorted(clash)} are both a dst and a src of another pair")
    return live


def _copy_program(kv_pool, index_pool, pairs: list[tuple[int, int]], shift: int):
    cores = _cores()
    tensors = (kv_pool, index_pool)
    flat = [v for pair in pairs for v in pair]
    flat += [0] * (2 * MAX_PAIRS - len(flat))  # fixed argument count: one program for every list length
    units = len(pairs) * (1 << (shift - 5))
    used = min(CORES, units)
    kernel = fp.reader_kernel(
        COPY,
        cores,
        [a for t in tensors for a in fp.accessor_args(t)],
        [
            (ttnn.CoreCoord(j, 0), [*(t.buffer_address() for t in tensors), len(pairs), shift, j, CORES, *flat])
            for j in range(CORES)
        ],
    )
    cbs = [fp.cb_descriptor(0, ttnn.uint32, 2 * UNIT_BYTES + ALIGN_SLACK, 1, cores)]
    block_bytes = len(pairs) * (1 << shift) * ROW_BYTES
    meta = fp.program_meta(
        NAME,
        "copy_blocks",
        1,
        partial=((kv_pool, block_bytes), (index_pool, len(pairs) * (1 << shift) * 64)),
        dram_bytes=block_bytes + len(pairs) * (1 << shift) * 64,
        cores=used,
    )
    return fp.run_program([kv_pool, index_pool], fp.program_descriptor([kernel], cbs=cbs), meta=meta)


def copy_blocks(kv_pool, index_pool, pairs, *, block_size: int):
    """Copy block ``src`` to block ``dst`` in both pools for every ``(src, dst)`` of ``pairs`` (``src ==
    ZERO_SOURCE``: zero-fill ``dst``; ``src == dst``: skipped); returns ``kv_pool`` (both pools updated in place)."""

    shift = _check_block_size(block_size)
    blocks = _check_pools(kv_pool, index_pool, block_size)
    live = _check_pairs(pairs, blocks)
    for start in range(0, len(live), MAX_PAIRS):
        _copy_program(kv_pool, index_pool, live[start : start + MAX_PAIRS], shift)
    return kv_pool


def zero_blocks(kv_pool, index_pool, block_ids, *, block_size: int):
    """Zero-fill the blocks ``block_ids`` of both pools."""

    return copy_blocks(kv_pool, index_pool, [(ZERO_SOURCE, int(b)) for b in block_ids], block_size=block_size)


def _state_pages(state, batch_index: int | None) -> tuple[int, int, int]:
    """(page bytes, first page, total bytes) of the run: the whole tensor or batch row ``batch_index``."""

    if not _in_dram_interleaved(state):
        raise ValueError("the state tensor must be DRAM interleaved")
    shape = tuple(state.shape)
    if state.layout == ttnn.TILE_LAYOUT:
        if state.dtype not in fp.TILE_BYTES:
            raise ValueError(f"unsupported state dtype {state.dtype}")
        padded = tuple(state.padded_shape)
        page_bytes = fp.TILE_BYTES[state.dtype]
        pages = math.prod(padded) // (fp.TILE * fp.TILE)
    else:
        if state.dtype not in fp.ELEMENT_BYTES:
            raise ValueError(f"unsupported state dtype {state.dtype}")
        page_bytes = shape[-1] * fp.ELEMENT_BYTES[state.dtype]
        pages = math.prod(shape[:-1])
    if not (page_bytes % ROW_BYTES == 0 or (ROW_BYTES % page_bytes == 0 and page_bytes % 64 == 0)):
        raise ValueError(f"the state page of {page_bytes} bytes must be a multiple of {ROW_BYTES} or divide it (x64)")
    first = 0
    if batch_index is not None:
        if not 0 <= batch_index < shape[0] or pages % shape[0]:
            raise ValueError(f"batch index {batch_index} does not address a page run of {shape}")
        pages //= shape[0]
        first = batch_index * pages
    return page_bytes, first, pages * page_bytes


def _state_copy(state, kv_pool, block_id: int, block_size: int, byte_offset: int, batch_index, *, to_block: bool):
    shift = _check_block_size(block_size)
    kv = tuple(kv_pool.shape)
    if (
        len(kv) != 4
        or kv[:2] != (1, 1)
        or kv[3] != KV_WIDTH
        or kv_pool.dtype != ttnn.bfloat16
        or kv_pool.layout != ttnn.ROW_MAJOR_LAYOUT
        or kv[2] % block_size
        or not _in_dram_interleaved(kv_pool)
    ):
        raise ValueError(f"kv_pool must be a DRAM ROW_MAJOR bf16 [1, 1, NB * {block_size}, {KV_WIDTH}], got {kv}")
    if not 0 <= block_id < kv[2] // block_size:
        raise ValueError(f"block_id {block_id} is outside the {kv[2] // block_size} blocks")
    if byte_offset < 0 or byte_offset % ROW_BYTES:
        raise ValueError(f"byte_offset must be a non-negative multiple of {ROW_BYTES}, got {byte_offset}")
    page_bytes, first_page, total = _state_pages(state, batch_index)
    if byte_offset + total > block_size * ROW_BYTES:
        raise ValueError(
            f"{total} bytes at offset {byte_offset} do not fit the block of {block_size * ROW_BYTES} bytes"
        )
    cores = _cores()
    tensors = (state, kv_pool)
    row0 = (block_id << shift) + byte_offset // ROW_BYTES
    groups = -(-total // (GROUP_ROWS * ROW_BYTES))
    kernel = fp.reader_kernel(
        STATE,
        cores,
        [a for t in tensors for a in fp.accessor_args(t)],
        [
            (
                ttnn.CoreCoord(j, 0),
                [*(t.buffer_address() for t in tensors), total, first_page, page_bytes, row0, j, CORES],
            )
            for j in range(CORES)
        ],
        defines=[] if to_block else [("STATE_FROM_POOL", "1")],
    )
    cbs = [fp.cb_descriptor(0, ttnn.uint32, GROUP_ROWS * ROW_BYTES + ALIGN_SLACK, 1, cores)]
    meta = fp.program_meta(
        NAME,
        "state_to_block" if to_block else "block_to_state",
        1,
        partial=((state, total), (kv_pool, total)),
        cores=min(CORES, groups),
    )
    io = [state, kv_pool] if to_block else [kv_pool, state]
    return fp.run_program(io, fp.program_descriptor([kernel], cbs=cbs), meta=meta)


def state_to_block(src, kv_pool, block_id: int, *, block_size: int, byte_offset: int = 0, src_batch_index=None):
    """Copy the raw bytes of ``src`` (or of its batch row ``src_batch_index``) into K/V pool block ``block_id`` from
    ``byte_offset``; returns ``kv_pool`` (updated in place)."""

    _state_copy(src, kv_pool, int(block_id), block_size, int(byte_offset), src_batch_index, to_block=True)
    return kv_pool


def block_to_state(kv_pool, dst, block_id: int, *, block_size: int, byte_offset: int = 0, dst_batch_index=None):
    """The reverse of ``state_to_block``: overwrite the pages of ``dst`` (or of its batch row ``dst_batch_index``) with
    the bytes of K/V pool block ``block_id`` from ``byte_offset``; returns ``dst`` (updated in place)."""

    _state_copy(dst, kv_pool, int(block_id), block_size, int(byte_offset), dst_batch_index, to_block=False)
    return dst

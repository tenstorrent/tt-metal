# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pure-torch references of the paged QSA KV mappings (opt-in PAGED mode of selection_row / main_tail and the
kv_page_scatter program).

Conventions: page size ``bs`` a power of two >= 32; page table int32 [32, max_blocks] (row u = lane u, rows past the
lanes zero, page 0 the reserved scratch page); physical row of lane u's token t = ``pt[u][t >> shift] * bs + (t & (bs - 1))``.
"""

from __future__ import annotations

import torch

MASKED = 0xFFFFFFFF


def physical_rows(page_row: torch.Tensor, tokens: torch.Tensor, page_size: int) -> torch.Tensor:
    """int64 physical rows of ``tokens`` (int64, any shape) through one page-table row."""

    shift = page_size.bit_length() - 1
    return page_row.to(torch.int64)[tokens >> shift] * page_size + (tokens & (page_size - 1))


def selection_row_paged_reference(values: torch.Tensor, page_table: torch.Tensor, page_size: int) -> torch.Tensor:
    """``values`` int64 [rows, W] (the finished selection row, lane-local token indices, 0xFFFFFFFF masked) ->
    the paged selection row: row r maps through ``page_table[r]``."""

    out = torch.empty_like(values)
    for r in range(values.shape[0]):
        masked = values[r] == MASKED
        safe = torch.where(masked, torch.zeros_like(values[r]), values[r])
        mapped = physical_rows(page_table[r], safe, page_size)
        out[r] = torch.where(masked, torch.full_like(mapped, MASKED), mapped)
    return out


def selection_row_reference(block_ids, offsets, keep, fill, sentinel_pad) -> torch.Tensor:
    """The unpaged selection row (int64 [rows, 2080]): ((ids << 2) + offsets repeated, sentinel tail) & keep | fill."""

    rows = block_ids.shape[0]
    expanded = (block_ids << 2).repeat_interleave(4, dim=1) + offsets.expand(rows, -1)
    template = torch.cat([expanded, sentinel_pad.expand(rows, -1)], dim=1) & 0xFFFFFFFF
    return (template & keep) | fill


def kv_block_base_reference(page_table: torch.Tensor, block_start: torch.Tensor, page_size: int) -> torch.Tensor:
    """main_tail's paged write base row per lane (``block_start`` int64 [lanes] = P & ~31)."""

    return torch.stack(
        [physical_rows(page_table[u], block_start[u : u + 1], page_size)[0] for u in range(block_start.shape[0])]
    )


def pool_after_block_writes(pool: torch.Tensor, blocks: torch.Tensor, bases: torch.Tensor) -> torch.Tensor:
    """``pool`` [rows, 512] with lane u's 32-row ``blocks[u]`` ([lanes, 32, 512]) at rows ``bases[u]``."""

    out = pool.clone()
    for u in range(blocks.shape[0]):
        out[int(bases[u]) : int(bases[u]) + 32] = blocks[u]
    return out


def pool_after_scatter(pool: torch.Tensor, src: torch.Tensor, page_row: torch.Tensor, rows: int, page_size: int):
    """``pool`` [R, 512] after kv_page_scatter: src rows [0, roundup(rows, 32)) at the page row's physical rows."""

    out = pool.clone()
    count = -(-rows // 32) * 32
    tokens = torch.arange(count, dtype=torch.int64)
    out[physical_rows(page_row, tokens, page_size)] = src[:count]
    return out


def pool_after_scatter_at(
    pool: torch.Tensor,
    src: torch.Tensor,
    page_table: torch.Tensor,
    lane: int,
    start: int,
    rows: int,
    page_size: int,
    src_row0: int = 0,
):
    """``pool`` [R, 512] after kv_page_scatter with a device position: src rows [src_row0, src_row0 + roundup(rows, 32))
    at the lane's logical rows [start, start + roundup(rows, 32))."""

    out = pool.clone()
    count = -(-rows // 32) * 32
    tokens = start + torch.arange(count, dtype=torch.int64)
    out[physical_rows(page_table[lane], tokens, page_size)] = src[src_row0 : src_row0 + count]
    return out


def random_page_table(lanes: int, max_blocks: int, generator: torch.Generator) -> tuple[torch.Tensor, int]:
    """(int32 [32, max_blocks] page table, num_pages): lanes' entries distinct shuffled pages 1 .. lanes * max_blocks
    (page 0 reserved scratch; rows past the lanes zero)."""

    pages = 1 + torch.randperm(lanes * max_blocks, generator=generator)
    table = torch.zeros(32, max_blocks, dtype=torch.int32)
    table[:lanes] = pages.reshape(lanes, max_blocks).to(torch.int32)
    return table, 1 + lanes * max_blocks


def index_pool_after_fill(
    pool: torch.Tensor,
    src_rows: torch.Tensor,
    page_table: torch.Tensor,
    lane: int,
    start_pos: int,
    block_rows: int,
) -> torch.Tensor:
    """``pool`` [NB, 1, block_rows, 128] after index_page_fill: ``src_rows`` [n * 32, 128] (n tile-rows) at logical
    tile-row ``start_pos >> 7`` + i -> physical tile-row ``pt[lane][L >> log2(tpb)] * tpb + (L & (tpb - 1))``,
    tpb = block_rows / 32.  Every other tile is unchanged."""

    tpb = block_rows // 32
    flat = pool.reshape(-1, 128).clone()
    for i in range(src_rows.shape[0] // 32):
        logical = (start_pos >> 7) + i
        row = int(page_table[lane][logical // tpb]) * tpb + logical % tpb
        flat[row * 32 : (row + 1) * 32] = src_rows[i * 32 : (i + 1) * 32]
    return flat.reshape(pool.shape)


def scatter_index_pool(
    logical_k: torch.Tensor,
    page_table: torch.Tensor,
    block_rows: int,
    num_blocks: int,
    generator=None,
    user_rows: int = None,
) -> torch.Tensor:
    """With ``user_rows`` (S <= W * block_rows) see scatter_index_pool_user_rows.
    Pool [num_blocks, 1, block_rows, 128] holding ``logical_k`` [R, 1, W * block_rows, 128] through
    ``page_table`` [R, W] (any shape with R * W entries, read flattened row-major): logical row j of batch b goes to
    block ptf[(b * W * block_rows + j) // block_rows], row (b * W * block_rows + j) % block_rows.  Blocks the table
    does not name (block 0 included) hold random finite values."""

    if user_rows is not None:
        return scatter_index_pool_user_rows(logical_k, page_table, block_rows, num_blocks, user_rows, generator)
    users, _, rows, width = logical_k.shape
    ptf = page_table.reshape(-1).to(torch.int64)
    pool = torch.randn(num_blocks, 1, block_rows, width, generator=generator).to(logical_k.dtype)
    flat = logical_k.reshape(users * rows, width)
    for index in range(users * rows // block_rows):
        pool[int(ptf[index]), 0] = flat[index * block_rows : (index + 1) * block_rows]
    return pool


def scatter_index_pool_user_rows(
    logical_k: torch.Tensor, page_table: torch.Tensor, block_rows: int, num_blocks: int, user_rows: int, generator=None
) -> torch.Tensor:
    """Pool [num_blocks, 1, block_rows, 128] holding ``logical_k`` [R, 1, user_rows, 128] through ``page_table``
    [R, W] with W * block_rows >= user_rows: logical row j of user b goes to block pt[b][j // block_rows], row
    j % block_rows.  Rows of a partly used last block, blocks the table does not name and block 0 hold random finite
    values."""

    users = logical_k.shape[0]
    width = page_table.reshape(users, -1).shape[1]
    ptf = page_table.reshape(users, width).to(torch.int64)
    pool = torch.randn(num_blocks, 1, block_rows, logical_k.shape[-1], generator=generator).to(logical_k.dtype)
    for user in range(users):
        for index in range((user_rows + block_rows - 1) // block_rows):
            chunk = logical_k[user, 0, index * block_rows : min(user_rows, (index + 1) * block_rows)]
            pool[int(ptf[user, index]), 0, : chunk.shape[0]] = chunk
    return pool


POOL_ZERO_SOURCE = 0xFFFFFFFF


def pool_after_copy_blocks(kv: torch.Tensor, index_tiles: torch.Tensor, pairs, block_size: int):
    """(kv, index_tiles) after ``pool_blocks.copy_blocks``: ``kv`` [1, 1, NB * bs, 512] (or [NB * bs, 512]) and
    ``index_tiles`` [NB, 1, bs / 4, 128]; block ``src`` goes to block ``dst`` in both (``src == POOL_ZERO_SOURCE``:
    zeros; ``src == dst``: nothing), all pairs read the pools as they were before the call.  Returns new tensors."""

    kv_out, index_out = kv.clone(), index_tiles.clone()
    rows = kv_out.reshape(-1, kv_out.shape[-1])
    source_rows = kv.reshape(-1, kv.shape[-1])
    for src, dst in pairs:
        if src == dst:
            continue
        window = slice(dst * block_size, (dst + 1) * block_size)
        if src == POOL_ZERO_SOURCE:
            rows[window] = 0
            index_out[dst] = 0
        else:
            rows[window] = source_rows[src * block_size : (src + 1) * block_size]
            index_out[dst] = index_tiles[src]
    return kv_out, index_out


def tile_order_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """The device byte stream (uint8) of a TILE tensor [..., H, W] (H, W multiples of 32) of a 2- or 4-byte dtype:
    tiles in row-major tile order over the leading dims, each tile as four 16 x 16 faces (top-left, top-right,
    bottom-left, bottom-right), each face row-major."""

    *lead, height, width = tensor.shape
    tiles = tensor.reshape(-1, height // 32, 2, 16, width // 32, 2, 16)  # n, th, face_row, row, tw, face_col, col
    tiles = tiles.permute(0, 1, 4, 2, 5, 3, 6)  # n, th, tw, face_row, face_col, row, col
    return tiles.contiguous().reshape(-1).view(torch.uint8)


def pool_after_state_to_block(
    kv_bytes: torch.Tensor, src_bytes: torch.Tensor, block: int, block_size: int, byte_offset: int
) -> torch.Tensor:
    """``kv_bytes`` (uint8, the K/V pool's bytes, 1024 per row) after ``pool_blocks.state_to_block``: ``src_bytes``
    (uint8, the state's pages in page order) at byte ``byte_offset`` of block ``block``."""

    out = kv_bytes.clone().reshape(-1)
    start = block * block_size * 1024 + byte_offset
    out[start : start + src_bytes.numel()] = src_bytes.reshape(-1)
    return out

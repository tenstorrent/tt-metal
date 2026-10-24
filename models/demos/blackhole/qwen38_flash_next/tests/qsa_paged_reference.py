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


def random_page_table(lanes: int, max_blocks: int, generator: torch.Generator) -> tuple[torch.Tensor, int]:
    """(int32 [32, max_blocks] page table, num_pages): lanes' entries distinct shuffled pages 1 .. lanes * max_blocks
    (page 0 reserved scratch; rows past the lanes zero)."""

    pages = 1 + torch.randperm(lanes * max_blocks, generator=generator)
    table = torch.zeros(32, max_blocks, dtype=torch.int32)
    table[:lanes] = pages.reshape(lanes, max_blocks).to(torch.int32)
    return table, 1 + lanes * max_blocks

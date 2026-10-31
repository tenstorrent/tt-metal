# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Page-table selection shared by Laguna's single-shot prefill paths."""

from __future__ import annotations

import ttnn


def single_shot_fill_page_table(
    fill_page_table,
    *,
    start_pos: int,
    seq_len: int,
    block_size: int,
    fill_page_table_base_pos: int = 0,
):
    """Return the fill table for one non-pipelined prefill.

    ``fill_page_table_base_pos`` is the absolute position represented by column
    zero. The vLLM adapter host-rebases each request row to its resumed start, so
    both cold and resumed serving prefills return the persistent full table without
    allocating a transient device slice. Direct callers retain the historical
    absolute-table behavior through the default base position of zero.
    """
    start_pos = int(start_pos)
    base_pos = int(fill_page_table_base_pos)
    seq_len = int(seq_len)
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"fill page-table block size must be positive, got {block_size}")
    if seq_len <= 0:
        raise ValueError(f"fill page-table sequence length must be positive, got {seq_len}")
    relative_start = start_pos - base_pos
    if relative_start < 0:
        raise ValueError(
            f"fill start_pos {start_pos} precedes page-table base position {base_pos}"
        )
    if relative_start % block_size:
        raise ValueError(
            f"fill relative start {relative_start} is not aligned to block size {block_size}"
        )
    if relative_start == 0:
        return fill_page_table

    col0 = relative_start // block_size
    ncol = (seq_len + block_size - 1) // block_size
    return ttnn.slice(
        fill_page_table,
        [0, col0],
        [fill_page_table.shape[0], col0 + ncol],
    )

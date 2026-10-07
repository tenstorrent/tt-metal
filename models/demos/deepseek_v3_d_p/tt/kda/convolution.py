# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compact predecessor and replacement carry exchange, with fixed collective participation."""

from functools import partial

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.chronological_selections import _layout as _selection_layout


def exchange_convolution_carry(
    projected_qkv: ttnn.Tensor,
    *,
    sequence_parallel_axis: int,
    actual_start: ttnn.Tensor,
    actual_end: ttnn.Tensor | None,
    local_rows: int,
    width: int,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Return predecessor history and the replacement logical stream carry.

    Both outputs are BF16 row-major DRAM tensors shaped ``[1, 3, local_channels]``. Predecessor history varies by
    SP rank; the final carry is replicated across each SP line. Channels remain sharded across TP. The native
    convolution selects the caller's initial history at the logical sequence start. ``projected_qkv`` is the tiled
    projection whose leading ``width`` columns are the channels. Every selection derives its rows on device from
    the bounds.
    """
    select = partial(
        ttnn.experimental.kda.select_history_rows,
        actual_start=actual_start,
        sequence_parallel_axis=sequence_parallel_axis,
        local_rows=local_rows,
        actual_end=actual_end,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # One selection packs the outgoing and the local final history as six rows per rank, so one gather along rows
    # carries both; one selection then splits the predecessor's outgoing and the final owner's local final rows.
    (histories,) = select(projected_qkv, _selection_layout.OUTGOING_AND_LOCAL_FINAL_HISTORY, width=width)
    # The six-row gather is latency-bound: the plain all_gather beats the high-bandwidth transport here.
    gathered = ttnn.all_gather(
        histories, dim=1, cluster_axis=sequence_parallel_axis, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    predecessor, final_carry = select(
        gathered, _selection_layout.PREDECESSOR_AND_FINAL_HISTORY, rows_per_output=_selection_layout.HISTORY_ROWS
    )
    return predecessor, final_carry

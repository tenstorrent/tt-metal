# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compact predecessor and replacement carry exchange, with fixed collective participation."""

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.chronological_selections import ChronologicalSelections
from models.demos.deepseek_v3_d_p.tt.kda.collectives import sp_all_gather


def exchange_convolution_carry(
    projected_qkv: ttnn.Tensor,
    *,
    sequence_parallel_axis: int,
    selections: ChronologicalSelections,
    width: int,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Return predecessor history and the replacement logical stream carry.

    Both outputs are BF16 row-major DRAM tensors. The predecessor is ``[1, 6, local_channels]``, whose leading
    three rows are the history; the final carry is ``[1, 3, local_channels]``. Predecessor history varies by SP
    rank; the final carry is replicated across each SP line. Channels remain sharded across TP. The native convolution
    selects the caller's initial history at the logical sequence start.
    ``projected_qkv`` is the tiled projection whose leading ``width`` columns are the channels.
    """
    # One selection packs the outgoing and the local final history as six rows per rank, so one gather along rows
    # carries both, and one selection picks the predecessor's outgoing then the final owner's local final rows.
    histories = selections.select_outgoing_and_local_final_history(projected_qkv, width=width)
    gathered = sp_all_gather(histories, name="histories", dim=1, cluster_axis=sequence_parallel_axis)
    selected = selections.select_predecessor_and_final_history(gathered)
    # The convolution reads only the leading three predecessor rows, so it takes the packed selection as is.
    rows = selected.shape[1] // 2
    final_carry = ttnn.slice(selected, (0, rows, 0), (1, 2 * rows, width), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return selected, final_carry

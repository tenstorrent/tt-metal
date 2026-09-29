# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compact predecessor and replacement carry exchange, with fixed collective participation."""

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.chronological_selections import ChronologicalSelections


def exchange_convolution_carry(
    projected_qkv: ttnn.Tensor,
    *,
    sequence_parallel_axis: int,
    selections: ChronologicalSelections,
    width: int,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Return predecessor history and the replacement logical stream carry.

    Both outputs are BF16 row-major DRAM tensors shaped ``[1, 3, local_channels]``.
    Predecessor history varies by SP rank; the final carry is replicated across
    each SP line. Channels remain sharded across TP. The native convolution
    selects the caller's initial history at the logical sequence start.
    ``projected_qkv`` is the tiled projection whose leading ``width`` columns are the channels.
    """
    outgoing = selections.select_outgoing_history(projected_qkv, width=width)
    physical_tail_history = selections.select_local_final_history(projected_qkv, width=width)
    # One collective carries both histories side by side: gathering along rows keeps the rank-major
    # [rank * 3 + row] layout both selections index, so each selection reads its own half.
    packed = ttnn.concat([outgoing, physical_tail_history], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gathered = ttnn.all_gather(
        packed, dim=1, cluster_axis=sequence_parallel_axis, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    predecessor = _columns(selections.select_predecessor_history(gathered), 0, width)
    final_carry = _columns(selections.select_final_history(gathered), width, 2 * width)
    return predecessor, final_carry


def _columns(tensor: ttnn.Tensor, start: int, end: int) -> ttnn.Tensor:
    batch, rows, _ = tensor.shape
    return ttnn.slice(tensor, (0, 0, start), (batch, rows, end), memory_config=ttnn.DRAM_MEMORY_CONFIG)

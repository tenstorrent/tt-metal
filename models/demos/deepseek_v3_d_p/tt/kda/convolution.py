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
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Return predecessor history and the replacement logical stream carry.

    Both outputs are BF16 row-major DRAM tensors shaped ``[1, 3, local_channels]``.
    Predecessor history varies by SP rank; the final carry is replicated across
    each SP line. Channels remain sharded across TP. The native convolution
    selects the caller's initial history at the logical sequence start.
    """
    outgoing = selections.select_outgoing_history(projected_qkv)
    gathered_outgoing_history = ttnn.all_gather(
        outgoing, dim=1, cluster_axis=sequence_parallel_axis, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    predecessor = selections.select_predecessor_history(gathered_outgoing_history)
    # A split rank sends its head's history to its successor, but its physical
    # tail supplies the final carry. Exchange physical tails separately so both
    # sources remain available to the runtime selections.
    batch, rows, width = projected_qkv.shape
    physical_tail_history = ttnn.slice(
        projected_qkv,
        (0, rows - outgoing.shape[1], 0),
        (batch, rows, width),
    )
    broadcast_tail_histories = ttnn.all_broadcast(physical_tail_history, cluster_axis=sequence_parallel_axis)
    candidates = ttnn.concat(broadcast_tail_histories, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    final_carry = selections.select_final_history(candidates)
    return predecessor, final_carry

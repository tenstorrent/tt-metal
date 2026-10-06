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
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Return predecessor history and the replacement logical stream carry.

    Both outputs are BF16 row-major DRAM tensors shaped ``[1, 3, local_channels]``.
    Predecessor history varies by SP rank; the final carry is replicated across
    each SP line. Channels remain sharded across TP. The native convolution
    selects the caller's initial history at the logical sequence start.
    """
    outgoing = selections.select_outgoing_history(projected_qkv)
    gathered_outgoing_history = sp_all_gather(
        outgoing, name="outgoing_history", dim=1, cluster_axis=sequence_parallel_axis
    )
    predecessor = selections.select_predecessor_history(gathered_outgoing_history)
    physical_tail_history = selections.select_local_final_history(
        projected_qkv, tuple(projected_qkv.device().shape)[sequence_parallel_axis]
    )
    candidates = sp_all_gather(physical_tail_history, name="tail_histories", dim=1, cluster_axis=sequence_parallel_axis)
    final_carry = selections.select_final_history(candidates)
    return predecessor, final_carry

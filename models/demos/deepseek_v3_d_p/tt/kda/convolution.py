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
    outgoing = selections.select_outgoing_history(projected_qkv)
    gathered = ttnn.all_gather(
        outgoing, dim=1, cluster_axis=sequence_parallel_axis, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    predecessor = selections.select_predecessor_history(gathered)
    batch, rows, width = projected_qkv.shape
    physical_end = ttnn.slice(projected_qkv, (0, rows - outgoing.shape[1], 0), (batch, rows, width))
    finals = ttnn.all_broadcast(physical_end, cluster_axis=sequence_parallel_axis)
    candidates = ttnn.concat(finals, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    final_carry = selections.select_final_history(candidates)
    return predecessor, final_carry

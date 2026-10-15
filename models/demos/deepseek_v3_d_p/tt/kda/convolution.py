# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compact predecessor and replacement carry exchange, with fixed collective participation."""

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.chronological_selections import ChronologicalSelections


def _has_l1_small(device) -> bool:
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1_SMALL)
    return view.total_bytes_allocated_per_bank + view.total_bytes_free_per_bank > 0


def exchange_convolution_carry(
    projected_qkv: ttnn.Tensor,
    layer_history: ttnn.Tensor,
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
    physical_tail_history = selections.select_local_final_history(projected_qkv, layer_history, predecessor)
    # all_broadcast's global semaphores live as long as its cached program and default to L1. The first
    # call runs while the layer's L1 input projection is live, so they would land just below it and,
    # once it frees, split L1 for good (ring_mla's static CBs then no longer fit). all_gather already
    # takes L1_SMALL whenever the device has one; match it.
    broadcast_tail_histories = ttnn.all_broadcast(
        physical_tail_history,
        cluster_axis=sequence_parallel_axis,
        use_l1_small_for_semaphores=_has_l1_small(projected_qkv.device()),
    )
    candidates = ttnn.concat(broadcast_tail_histories, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    final_carry = selections.select_final_history(candidates)
    return predecessor, final_carry

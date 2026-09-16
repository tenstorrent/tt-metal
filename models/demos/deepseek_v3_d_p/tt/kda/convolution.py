# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel carry exchange for KDA causal convolution."""

from __future__ import annotations

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.chronological_topology import ChronologicalTopology


def exchange_convolution_carry(
    projected_qkv: ttnn.Tensor,
    initial_carry: ttnn.Tensor,
    *,
    sequence_parallel_axis: int,
    topology: ChronologicalTopology,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Return partition entry histories and the replicated final stream carry.

    Inputs are read only. Outputs are BF16 row-major DRAM tensors with TP-sharded
    channels. Each SP rank receives three entry rows, or six for a split: the
    first plane seeds its physical head and the second seeds the boundary tail.
    The three-row final carry is replicated across SP.

    Only three-row histories cross the mesh. Unsplit execution reuses its halo
    gather for the final carry; split execution separately publishes the boundary
    rank's physical end, which differs from its outgoing head halo.
    """
    batch, local_rows, channels = projected_qkv.shape
    history = initial_carry.shape[1]
    axis = sequence_parallel_axis

    def rows(tensor: ttnn.Tensor, start: int) -> ttnn.Tensor:
        return ttnn.slice(
            tensor,
            (0, start, 0),
            (batch, start + history, channels),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    physical_end = rows(projected_qkv, local_rows - history)
    if topology.sp_size == 1:
        return initial_carry, physical_end

    outgoing = physical_end
    if topology.is_split:
        head_end = rows(projected_qkv, topology.head_rows - history)
        outgoing = ttnn.mesh_partition(
            ttnn.concat(
                [head_end if rank == topology.boundary_chip else physical_end for rank in range(topology.sp_size)],
                dim=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            dim=1,
            cluster_axis=axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    gathered = ttnn.all_gather(outgoing, dim=1, cluster_axis=axis, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tails = [rows(gathered, rank * history) for rank in range(topology.sp_size)]
    entries = []
    for rank in range(topology.sp_size):
        predecessor = tails[topology.predecessor_chip(rank)]
        entry = initial_carry if rank == topology.boundary_chip else predecessor
        entries.append(entry)
        if topology.is_split:
            entries.append(predecessor if rank == topology.boundary_chip else entry)

    if topology.is_split:
        # all_broadcast supplies a source on every TP line; a single global mesh
        # coordinate cannot express these independent senders.
        finals = ttnn.all_broadcast(physical_end, cluster_axis=axis)
        final_carry = finals[topology.boundary_chip]
        for rank, tensor in enumerate(finals):
            if rank != topology.boundary_chip:
                ttnn.deallocate(tensor)
    else:
        final_carry = tails[topology.chip_order[-1]]
    partition_carry = ttnn.mesh_partition(
        ttnn.concat(entries, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        dim=1,
        cluster_axis=axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return partition_carry, final_carry

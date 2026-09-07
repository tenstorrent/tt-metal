# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel carry exchange for KDA causal convolution."""

from __future__ import annotations

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.offset import OffsetTopology


def exchange_convolution_carry(
    projected_qkv: ttnn.Tensor,
    initial_carry: ttnn.Tensor,
    *,
    sequence_parallel_axis: int,
    topology: OffsetTopology,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Return partition entry carries and the replicated final stream carry.

    Both outputs have shape ``[B, history, Q_local + K_local + V_local]`` in
    row-major DRAM. ``partition_carry`` differs by SP rank: the chronologically
    first chip receives ``initial_carry`` and every other chip receives its
    chronological predecessor's tail. ``final_carry`` is the global stream tail
    replicated across SP. Channels remain sharded across TP.

    Order comes from ``topology``, never from physical rank. At ``actual_start=0``
    the boundary chip is rank zero and this reduces exactly to rank order.
    """
    batch, local_sequence, channels = projected_qkv.shape
    history = initial_carry.shape[1]
    mesh_device = projected_qkv.device()
    mesh_shape = tuple(mesh_device.shape)
    sp_size = mesh_shape[sequence_parallel_axis]

    local_tail = ttnn.slice(
        projected_qkv,
        (0, local_sequence - history, 0),
        (batch, local_sequence, channels),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    padded_tail = ttnn.pad(
        local_tail,
        ((0, 0), (0, ttnn.TILE_SIZE - history), (0, 0)),
        value=0.0,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tiled_tail = ttnn.to_layout(padded_tail, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gathered_tails = ttnn.all_gather(
        tiled_tail,
        dim=1,
        cluster_axis=sequence_parallel_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def chip_tail(chip: int) -> ttnn.Tensor:
        tiled_chip_tail = ttnn.slice(
            gathered_tails,
            (0, chip * ttnn.TILE_SIZE, 0),
            (batch, (chip + 1) * ttnn.TILE_SIZE, channels),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        row_major_tail = ttnn.to_layout(tiled_chip_tail, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.slice(
            row_major_tail,
            (0, 0, 0),
            (batch, history, channels),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Chronological order: the boundary chip opens the stream and consumes the
    # caller carry; every other chip consumes the tail of the chip before it.
    entry_carries = [
        initial_carry if chip == topology.boundary_chip else chip_tail(topology.predecessor_chip(chip))
        for chip in range(sp_size)
    ]
    replicated_entries = ttnn.concat(entry_carries, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    partition_carry = ttnn.mesh_partition(
        replicated_entries,
        dim=1,
        cluster_axis=sequence_parallel_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # The stream ends on the chip chronologically before the boundary chip.
    final_carry = chip_tail(topology.predecessor_chip(topology.boundary_chip))
    return partition_carry, final_carry

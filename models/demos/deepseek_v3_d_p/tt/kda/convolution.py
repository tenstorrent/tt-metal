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
    wrap_indicator: ttnn.Tensor | None = None,
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

    def local_tail(end: int) -> ttnn.Tensor:
        return ttnn.slice(
            projected_qkv,
            (0, end - history, 0),
            (batch, end, channels),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    if topology.is_split:
        if wrap_indicator is None:
            raise ValueError("a split convolution carry exchange requires a device-local wrap indicator")

        # One device program selects the source address locally. Ordinary ranks
        # read only their physical end; the wrap rank publishes its head end and
        # retains its physical end in rows [history:2*history] of the same tile.
        tiled_tail = ttnn.experimental.kda.pack_convolution_carry(
            projected_qkv,
            wrap_indicator,
            topology.head_rows,
            history_rows=history,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        physical_end = local_tail(local_sequence)
        padded_tail = ttnn.pad(
            physical_end,
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

    def chip_tail(chip: int, *, retained_state: bool = False) -> ttnn.Tensor:
        row = history if retained_state else 0
        tiled_chip_tail = ttnn.slice(
            gathered_tails,
            (0, chip * ttnn.TILE_SIZE, 0),
            (batch, (chip + 1) * ttnn.TILE_SIZE, channels),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        row_major_tail = ttnn.to_layout(tiled_chip_tail, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.slice(
            row_major_tail,
            (0, row, 0),
            (batch, row + history, channels),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    entry_carries = []
    for chip in range(sp_size):
        entry = initial_carry if chip == topology.boundary_chip else chip_tail(topology.predecessor_chip(chip))
        entry_carries.append(entry)
        if topology.is_split:
            # Only the wrap rank reads plane 1, at its tail. Its predecessor is
            # the last ordinary rank in chronological order. Duplicating plane 0
            # elsewhere keeps the mesh tensor uniform without inventing a split.
            tail_entry = chip_tail(topology.predecessor_chip(chip)) if chip == topology.boundary_chip else entry
            entry_carries.append(tail_entry)

    final_carry = (
        chip_tail(topology.boundary_chip, retained_state=True)
        if topology.is_split
        else chip_tail(topology.chip_order[-1])
    )
    replicated_entries = ttnn.concat(entry_carries, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    partition_carry = ttnn.mesh_partition(
        replicated_entries,
        dim=1,
        cluster_axis=sequence_parallel_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return partition_carry, final_carry

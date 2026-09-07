# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel carry exchange for KDA causal convolution."""

from __future__ import annotations

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.offset import OffsetTopology, fragment_order


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

    # One tile slot per fragment end. When the offset splits the partition each
    # chip contributes both its head end and its tail end, because a chip's head
    # feeds its own tail while its tail feeds the next chip.
    fragment_ends = [topology.head_rows, local_sequence] if topology.is_split else [local_sequence]
    slots_per_chip = len(fragment_ends)
    padded_tails = []
    for end in fragment_ends:
        local_tail = ttnn.slice(
            projected_qkv,
            (0, end - history, 0),
            (batch, end, channels),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        padded_tails.append(
            ttnn.pad(
                local_tail,
                ((0, 0), (0, ttnn.TILE_SIZE - history), (0, 0)),
                value=0.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
    padded_tail = (
        padded_tails[0]
        if slots_per_chip == 1
        else ttnn.concat(padded_tails, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    )
    tiled_tail = ttnn.to_layout(padded_tail, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gathered_tails = ttnn.all_gather(
        tiled_tail,
        dim=1,
        cluster_axis=sequence_parallel_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def chip_tail(chip: int, fragment: int = 0) -> ttnn.Tensor:
        slot = chip * slots_per_chip + fragment
        tiled_chip_tail = ttnn.slice(
            gathered_tails,
            (0, slot * ttnn.TILE_SIZE, 0),
            (batch, (slot + 1) * ttnn.TILE_SIZE, channels),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        row_major_tail = ttnn.to_layout(tiled_chip_tail, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.slice(
            row_major_tail,
            (0, 0, 0),
            (batch, history, channels),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Walk the canonical chronological order rather than re-deriving predecessors.
    # The boundary chip's head opens the stream and its tail closes it, so the
    # chip after the boundary takes the boundary chip's HEAD end, not its tail --
    # a rule that is easy to get wrong when written out by hand.
    order = fragment_order(topology) if topology.is_split else tuple((chip, 0) for chip in topology.chip_order)
    entry_by_slot: dict[tuple[int, int], ttnn.Tensor] = {}
    previous: tuple[int, int] | None = None
    for slot in order:
        entry_by_slot[slot] = initial_carry if previous is None else chip_tail(*previous)
        previous = slot
    assert previous is not None
    final_carry = chip_tail(*previous)

    entry_carries = [entry_by_slot[(chip, fragment)] for chip in range(sp_size) for fragment in range(slots_per_chip)]
    replicated_entries = ttnn.concat(entry_carries, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    partition_carry = ttnn.mesh_partition(
        replicated_entries,
        dim=1,
        cluster_axis=sequence_parallel_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return partition_carry, final_carry

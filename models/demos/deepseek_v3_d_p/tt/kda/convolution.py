# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel carry exchange for KDA causal convolution."""

from __future__ import annotations

import ttnn


def exchange_convolution_carry(
    projected_qkv: ttnn.Tensor,
    initial_carry: ttnn.Tensor,
    *,
    sequence_parallel_axis: int,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Return partition entry carries and the replicated final stream carry.

    Both outputs have shape ``[B, history, Q_local + K_local + V_local]`` in
    row-major DRAM. ``partition_carry`` differs by SP rank: rank zero receives
    ``initial_carry`` and every later rank receives its predecessor tail.
    ``final_carry`` is the global stream tail replicated across SP. Channels
    remain sharded across TP.
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

    entry_carries = [initial_carry]
    for rank in range(sp_size - 1):
        tiled_rank_tail = ttnn.slice(
            gathered_tails,
            (0, rank * ttnn.TILE_SIZE, 0),
            (batch, (rank + 1) * ttnn.TILE_SIZE, channels),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rank_tail = ttnn.to_layout(tiled_rank_tail, ttnn.ROW_MAJOR_LAYOUT)
        entry_carries.append(
            ttnn.slice(
                rank_tail,
                (0, 0, 0),
                (batch, history, channels),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
    replicated_entries = ttnn.concat(entry_carries, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    partition_carry = ttnn.mesh_partition(
        replicated_entries,
        dim=1,
        cluster_axis=sequence_parallel_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tiled_final_carry = ttnn.slice(
        gathered_tails,
        (0, (sp_size - 1) * ttnn.TILE_SIZE, 0),
        (batch, sp_size * ttnn.TILE_SIZE, channels),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    final_row_major = ttnn.to_layout(tiled_final_carry, ttnn.ROW_MAJOR_LAYOUT)
    final_carry = ttnn.slice(
        final_row_major,
        (0, 0, 0),
        (batch, history, channels),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return partition_carry, final_carry

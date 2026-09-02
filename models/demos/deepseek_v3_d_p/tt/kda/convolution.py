# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel carry exchange for KDA causal convolution."""

from __future__ import annotations

import ttnn


def _gather_tails_at(
    tensor: ttnn.Tensor,
    end: int,
    history: int,
    *,
    sequence_parallel_axis: int,
) -> ttnn.Tensor:
    """Gather one padded convolution halo per physical SP rank."""
    batch, _, channels = tensor.shape
    local_tail = ttnn.slice(
        tensor,
        (0, end - history, 0),
        (batch, end, channels),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    padded_tail = ttnn.pad(
        local_tail,
        ((0, 0), (0, ttnn.TILE_SIZE - history), (0, 0)),
        value=0.0,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.all_gather(
        ttnn.to_layout(padded_tail, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        dim=1,
        cluster_axis=sequence_parallel_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _rank_tail(gathered_tails: ttnn.Tensor, rank: int, history: int) -> ttnn.Tensor:
    batch, _, channels = gathered_tails.shape
    tiled = ttnn.slice(
        gathered_tails,
        (0, rank * ttnn.TILE_SIZE, 0),
        (batch, (rank + 1) * ttnn.TILE_SIZE, channels),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    row_major = ttnn.to_layout(tiled, ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.slice(
        row_major,
        (0, 0, 0),
        (batch, history, channels),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _partition_rank_values(values: list[ttnn.Tensor], *, sequence_parallel_axis: int) -> ttnn.Tensor:
    replicated = ttnn.concat(values, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.mesh_partition(
        replicated,
        dim=1,
        cluster_axis=sequence_parallel_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def exchange_rotated_convolution_carry_prototype(
    projected_qkv: ttnn.Tensor,
    initial_carry: ttnn.Tensor,
    *,
    actual_start: int,
    sequence_parallel_axis: int,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Rotate rank-level halos for an offset exactly on an SP boundary."""
    _, local_sequence, _ = projected_qkv.shape
    history = initial_carry.shape[1]
    sp_size = tuple(projected_qkv.device().shape)[sequence_parallel_axis]
    boundary_rank = (actual_start // local_sequence) % sp_size
    gathered = _gather_tails_at(
        projected_qkv,
        local_sequence,
        history,
        sequence_parallel_axis=sequence_parallel_axis,
    )
    full_tail = [_rank_tail(gathered, rank, history) for rank in range(sp_size)]
    entries = [initial_carry if rank == boundary_rank else full_tail[(rank - 1) % sp_size] for rank in range(sp_size)]
    return (
        _partition_rank_values(entries, sequence_parallel_axis=sequence_parallel_axis),
        full_tail[(boundary_rank - 1) % sp_size],
    )


def exchange_split_convolution_carries_prototype(
    projected_qkv: ttnn.Tensor,
    initial_carry: ttnn.Tensor,
    *,
    actual_start: int,
    sequence_parallel_axis: int,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, int]:
    """Return head/tail entry carries for the sequential-tail prototype.

    The activation stays in MLA physical placement. Only two three-row halo
    candidates per SP rank are gathered. All ranks split at the boundary
    rank's row so existing equal-shape convolution programs can run twice;
    for non-boundary ranks the two pieces remain causally adjacent.
    """
    _, local_sequence, _ = projected_qkv.shape
    history = initial_carry.shape[1]
    mesh_shape = tuple(projected_qkv.device().shape)
    sp_size = mesh_shape[sequence_parallel_axis]
    split = actual_start % local_sequence
    if split == 0:
        raise ValueError("split carry exchange requires an in-device wrap")
    head_length = local_sequence - split
    boundary_rank = (actual_start // local_sequence) % sp_size

    head_tails = _gather_tails_at(
        projected_qkv,
        head_length,
        history,
        sequence_parallel_axis=sequence_parallel_axis,
    )
    full_tails = _gather_tails_at(
        projected_qkv,
        local_sequence,
        history,
        sequence_parallel_axis=sequence_parallel_axis,
    )
    head_tail = [_rank_tail(head_tails, rank, history) for rank in range(sp_size)]
    full_tail = [_rank_tail(full_tails, rank, history) for rank in range(sp_size)]

    head_entries: list[ttnn.Tensor] = []
    tail_entries: list[ttnn.Tensor] = []
    for rank in range(sp_size):
        if rank == boundary_rank:
            head_entries.append(initial_carry)
            tail_entries.append(full_tail[(rank - 1) % sp_size])
        else:
            predecessor = (rank - 1) % sp_size
            head_entries.append(head_tail[boundary_rank] if predecessor == boundary_rank else full_tail[predecessor])
            tail_entries.append(head_tail[rank])

    return (
        _partition_rank_values(head_entries, sequence_parallel_axis=sequence_parallel_axis),
        _partition_rank_values(tail_entries, sequence_parallel_axis=sequence_parallel_axis),
        full_tail[boundary_rank],
        head_length,
    )


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

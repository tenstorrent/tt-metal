# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Throwaway full-activation resharding prototype for offset KDA prefill."""

from __future__ import annotations

import ttnn

_PROTOTYPE_GLOBAL_SEQUENCE = 5120


def offset_segments(actual_start: int, sp_size: int, local_sequence: int) -> tuple[tuple[int, int, int], ...]:
    """Return ``(SP rank, local begin, local end)`` segments in causal order."""
    if actual_start < 0 or actual_start % ttnn.TILE_SIZE:
        raise ValueError(f"actual_start must be a non-negative multiple of {ttnn.TILE_SIZE}, got {actual_start}")
    if sp_size <= 0 or local_sequence <= 0 or local_sequence % ttnn.TILE_SIZE:
        raise ValueError("SP size and tile-aligned local sequence must be positive")

    boundary_rank = (actual_start // local_sequence) % sp_size
    tail_length = actual_start % local_sequence
    if tail_length == 0:
        return tuple(((boundary_rank + step) % sp_size, 0, local_sequence) for step in range(sp_size))

    head_length = local_sequence - tail_length
    middle = tuple(((boundary_rank + step) % sp_size, 0, local_sequence) for step in range(1, sp_size))
    return ((boundary_rank, 0, head_length), *middle, (boundary_rank, head_length, local_sequence))


def _slice_sequence(tensor: ttnn.Tensor, begin: int, end: int) -> ttnn.Tensor:
    return ttnn.slice(
        tensor,
        (0, begin, 0),
        (tensor.shape[0], end, tensor.shape[2]),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _concat_sequence(parts: list[ttnn.Tensor]) -> ttnn.Tensor:
    if len(parts) == 1:
        return parts[0]
    return ttnn.concat(parts, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def mla_to_temporal_sp(
    tensor: ttnn.Tensor,
    *,
    actual_start: int,
    sequence_parallel_axis: int,
) -> ttnn.Tensor:
    """Reshard MLA physical rows into equal chronological SP partitions."""
    mesh_device = tensor.device()
    sp_size = tuple(mesh_device.shape)[sequence_parallel_axis]
    local_sequence = tensor.shape[1]
    if local_sequence * sp_size != _PROTOTYPE_GLOBAL_SEQUENCE:
        raise ValueError(
            f"offset prototype requires {_PROTOTYPE_GLOBAL_SEQUENCE} global tokens, "
            f"got {local_sequence} local tokens across SP{sp_size}"
        )
    segments = offset_segments(actual_start, sp_size, local_sequence)
    if actual_start % _PROTOTYPE_GLOBAL_SEQUENCE == 0:
        return tensor

    physical = ttnn.all_gather(
        tensor,
        dim=1,
        cluster_axis=sequence_parallel_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    chronological_parts = [
        _slice_sequence(physical, rank * local_sequence + begin, rank * local_sequence + end)
        for rank, begin, end in segments
    ]
    chronological = _concat_sequence(chronological_parts)
    return ttnn.mesh_partition(
        chronological,
        dim=1,
        cluster_axis=sequence_parallel_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def temporal_to_mla_sp(
    tensor: ttnn.Tensor,
    *,
    actual_start: int,
    sequence_parallel_axis: int,
) -> ttnn.Tensor:
    """Restore chronological KDA output to MLA's physical SP row placement."""
    mesh_device = tensor.device()
    sp_size = tuple(mesh_device.shape)[sequence_parallel_axis]
    local_sequence = tensor.shape[1]
    if local_sequence * sp_size != _PROTOTYPE_GLOBAL_SEQUENCE:
        raise ValueError(
            f"offset prototype requires {_PROTOTYPE_GLOBAL_SEQUENCE} global tokens, "
            f"got {local_sequence} local tokens across SP{sp_size}"
        )
    segments = offset_segments(actual_start, sp_size, local_sequence)
    if actual_start % _PROTOTYPE_GLOBAL_SEQUENCE == 0:
        return tensor

    chronological = ttnn.all_gather(
        tensor,
        dim=1,
        cluster_axis=sequence_parallel_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    rank_parts: list[list[ttnn.Tensor]] = [[] for _ in range(sp_size)]
    cursor = 0
    for rank, begin, end in segments:
        length = end - begin
        rank_parts[rank].append(_slice_sequence(chronological, cursor, cursor + length))
        cursor += length
    physical_ranks = [_concat_sequence(parts) for parts in rank_parts]
    physical = _concat_sequence(physical_ranks)
    return ttnn.mesh_partition(
        physical,
        dim=1,
        cluster_axis=sequence_parallel_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

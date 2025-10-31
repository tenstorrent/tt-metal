# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import ttnn

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import torch


def bf16_tensor(
    x: torch.Tensor, device: ttnn.Device | None = None, mesh_axis=None, shard_dim=None, layout=ttnn.TILE_LAYOUT
) -> ttnn.Tensor:
    """
    Replicates or shards a tensor based on the mesh_axis and shard_dim
    """
    assert (mesh_axis is None) == (shard_dim is None)
    mesh_mapper = None
    if mesh_axis is not None:
        mapper_dims = [None, None]
        mapper_dims[mesh_axis] = shard_dim
        mesh_mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=mapper_dims)

    return ttnn.from_torch(
        x,
        layout=layout,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
        mesh_mapper=mesh_mapper,
    )


def bf16_tensor_host(
    x: torch.Tensor, device: ttnn.Device | None = None, mesh_axis=None, shard_dim=None, layout=ttnn.TILE_LAYOUT
) -> ttnn.Tensor:
    assert (mesh_axis is None) == (shard_dim is None)
    mesh_mapper = None
    if mesh_axis is not None:
        mapper_dims = [None, None]
        mapper_dims[mesh_axis] = shard_dim
        mesh_mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=mapper_dims)

    return ttnn.from_torch(
        x,
        layout=layout,
        dtype=ttnn.bfloat16,
        mesh_mapper=mesh_mapper,
    )


def bf16_tensor_2dshard(
    x: torch.Tensor, device: ttnn.Device, shard_mapping: dict[int, int], layout=ttnn.TILE_LAYOUT
) -> ttnn.Tensor:
    assert len(shard_mapping) == 2
    assert all(0 <= k <= 1 and 0 <= v < len(x.shape) for k, v in shard_mapping.items())
    mapper_dims = [None, None]
    for k, v in shard_mapping.items():
        mapper_dims[k] = v
    mesh_mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=mapper_dims)
    return ttnn.from_torch(
        x,
        layout=layout,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
        mesh_mapper=mesh_mapper,
    )


def from_torch(
    x: torch.Tensor,
    /,
    *,
    device: ttnn.MeshDevice | None = None,
    layout: ttnn.Layout = ttnn.Layout.TILE,
    dtype: ttnn.DataType = ttnn.bfloat16,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    pad_value: float | None = None,
    mesh_axes: Sequence[int | None] | None = None,
    on_host: bool = False,
) -> ttnn.Tensor:
    if mesh_axes is not None:
        if device is None:
            msg = "device must be specified if mesh_axes is given"
            raise ValueError(msg)

        mesh_rank = len(list(device.shape))
        verify_tensor_mesh_axes(mesh_axes, tensor_rank=len(x.shape), mesh_rank=mesh_rank)

        placements = _invert_placements(mesh_axes, output_rank=mesh_rank)
        placements = [ttnn.PlacementShard(p) if p is not None else ttnn.PlacementReplicate() for p in placements]
        mesh_mapper = ttnn.create_mesh_mapper(device, ttnn.MeshMapperConfig(placements))
    else:
        mesh_mapper = None

    return ttnn.from_torch(
        x,
        layout=layout,
        dtype=dtype,
        memory_config=memory_config,
        device=None if on_host else device,
        mesh_mapper=mesh_mapper,
        pad_value=pad_value,
    )


def to_torch(
    x: ttnn.Tensor,
    /,
    *,
    mesh_axes: Sequence[int | None] | None = None,
    composer_device: ttnn.MeshDevice | None = None,
) -> torch.Tensor:
    """Converts a ttnn.Tensor to a torch.Tensor.

    Strips away redundant data returned by calling ttnn.to_torch on a replicated tensor. If the
    tensor is distributed and not on device, composer_device must be provided.
    """
    if mesh_axes is None:
        if x.tensor_topology().distribution_shape() != ttnn.MeshShape([1]):
            msg = "mesh_axes must be specified for distributed tensors"
            raise ValueError(msg)
        return ttnn.to_torch(x)

    composer_device = composer_device or x.device()
    if composer_device is None:
        msg = "composer_device must be specified for distributed host tensors"
        raise ValueError(msg)

    mesh_rank = len(list(composer_device.shape))
    verify_tensor_mesh_axes(mesh_axes, tensor_rank=len(x.shape), mesh_rank=mesh_rank)

    replicated_mesh_axes = list(set(range(mesh_rank)) - {axis for axis in mesh_axes if axis is not None})
    mesh_axes = replicated_mesh_axes + list(mesh_axes)

    placements = _invert_placements(mesh_axes, output_rank=mesh_rank)
    assert all(p is not None for p in placements)

    mesh_composer = ttnn.create_mesh_composer(composer_device, ttnn.MeshComposerConfig(placements))

    x = x.reshape([1] * len(replicated_mesh_axes) + list(x.shape))
    return ttnn.to_torch(x, mesh_composer=mesh_composer)[(0,) * len(replicated_mesh_axes)]


def verify_tensor_mesh_axes(mesh_axes: Sequence[int | None], /, *, tensor_rank: int, mesh_rank: int) -> None:
    if len(mesh_axes) != tensor_rank:
        msg = f"mesh axis list {tuple(mesh_axes)} should have length {tensor_rank}"
        raise ValueError(msg)

    for axis in mesh_axes:
        if axis is not None and (axis < 0 or axis >= mesh_rank):
            msg = f"all mesh axes in mesh axis list {tuple(mesh_axes)} should be positive and smaller than {mesh_rank}"
            raise ValueError(msg)

    non_none_values = [p for p in mesh_axes if p is not None]
    if len(non_none_values) != len(set(non_none_values)):
        msg = f"mesh axis list {tuple(mesh_axes)} contains duplicate mesh axis assignments"
        raise ValueError(msg)


def _invert_placements(placements: Sequence[int | None], *, output_rank: int) -> tuple[int | None, ...]:
    out = [None] * output_rank

    for i, p in enumerate(placements):
        if p is not None:
            out[p] = i

    return tuple(out)


def create_mesh_mapper(mapping: Mapping[int | None, int | None], *, device: ttnn.MeshDevice) -> ttnn.CppTensorToMesh:
    mesh_rank = len(list(device.shape))

    placements = [ttnn.PlacementReplicate()] * mesh_rank

    for k, v in mapping.items():
        if k is None or v is None:
            continue
        assert k < mesh_rank, f"mesh mapping keys should be smaller than {mesh_rank}, got {k}"
        placements[k] = ttnn.PlacementShard(v)

    return ttnn.create_mesh_mapper(device, ttnn.MeshMapperConfig(placements))


def create_mesh_composer(mapping: Mapping[int | None, int | None], *, device: ttnn.MeshDevice) -> ttnn.CppMeshToTensor:
    mesh_rank = len(list(device.shape))

    placements = [0] * mesh_rank

    for k, v in mapping.items():
        if k is None or v is None:
            continue
        assert k < mesh_rank, f"mesh mapping keys should be smaller than {mesh_rank}, got {k}"
        placements[k] = v

    return ttnn.create_mesh_composer(device, ttnn.MeshComposerConfig(placements))

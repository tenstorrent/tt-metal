# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import ttnn

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch


def bf16_tensor(
    x: torch.Tensor, device: ttnn.Device | None = None, mesh_axis=None, shard_dim=None, layout=ttnn.TILE_LAYOUT
) -> ttnn.Tensor:
    """
    Replicates or shards a tensor based on the mesh_axis and shard_dim
    """
    assert (mesh_axis is None) == (shard_dim is None)
    return from_torch(x, device=device, layout=layout, mesh_mapping={mesh_axis: shard_dim})


def bf16_tensor_host(
    x: torch.Tensor, device: ttnn.Device | None = None, mesh_axis=None, shard_dim=None, layout=ttnn.TILE_LAYOUT
) -> ttnn.Tensor:
    assert (mesh_axis is None) == (shard_dim is None)
    return from_torch(x, device=device, layout=layout, mesh_mapping={mesh_axis: shard_dim}, to_host=True)


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
    device: ttnn.MeshDevice,
    layout: ttnn.Layout = ttnn.Layout.TILE,
    dtype: ttnn.DataType = ttnn.bfloat16,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapping: Mapping[int, int] | None = None,
    to_host: bool = False,
) -> ttnn.Tensor:
    mesh_mapper = create_mesh_mapper(mesh_mapping or {}, device=device)

    return ttnn.from_torch(
        x,
        layout=layout,
        dtype=dtype,
        memory_config=memory_config,
        device=None if to_host else device,
        mesh_mapper=mesh_mapper,
    )


def to_torch(
    x: ttnn.Tensor, /, *, device: ttnn.MeshDevice, mesh_mapping: Mapping[int, int] | None = None
) -> torch.Tensor:
    first_size = x.shape[0]
    mesh_composer = create_mesh_composer(mesh_mapping or {}, device=device)

    torch_x = ttnn.to_torch(x, mesh_composer=mesh_composer)
    return torch_x[:first_size]


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

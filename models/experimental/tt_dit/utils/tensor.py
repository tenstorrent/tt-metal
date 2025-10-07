# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import ttnn

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

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
    return from_torch(x, device=device, layout=layout, mesh_mapping={mesh_axis: shard_dim}, on_host=True)


def bf16_tensor_2dshard(
    x: torch.Tensor, device: ttnn.Device, shard_mapping: dict[int, int], layout=ttnn.TILE_LAYOUT
) -> ttnn.Tensor:
    assert len(shard_mapping) == 2
    return from_torch(x, device=device, layout=layout, mesh_mapping=shard_mapping)


def from_torch(
    x: torch.Tensor,
    /,
    *,
    device: ttnn.MeshDevice,
    layout: ttnn.Layout = ttnn.Layout.TILE,
    dtype: ttnn.DataType = ttnn.bfloat16,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapping: Mapping[int, int] | Mapping[int | None, int | None] | None = None,
    on_host: bool = False,
) -> ttnn.Tensor:
    mesh_rank = len(list(device.shape))
    mesh_placement = mesh_mapping_to_placement(mesh_mapping, mesh_rank=mesh_rank)

    return ttnn.from_torch(
        x,
        layout=layout,
        dtype=dtype,
        memory_config=memory_config,
        device=None if on_host else device,
        mesh_mapper=create_mesh_mapper(device, mesh_placement),
    )


def to_torch(
    x: ttnn.Tensor,
    /,
    *,
    device: ttnn.MeshDevice,
    mesh_mapping: Mapping[int, int] | Mapping[int | None, int | None] | None = None,
) -> torch.Tensor:
    mesh_rank = len(list(device.shape))
    mesh_placement = mesh_mapping_to_placement(mesh_mapping, mesh_rank=mesh_rank)

    size0 = x.shape[0]
    return ttnn.to_torch(x, mesh_composer=create_mesh_composer(device, mesh_placement))[:size0]


def create_mesh_mapper(device: ttnn.MeshDevice, placements: Iterable[int]) -> ttnn.CppTensorToMesh:
    placements = [ttnn.PlacementShard(p) if p is not None else ttnn.PlacementReplicate() for p in placements]
    return ttnn.create_mesh_mapper(device, ttnn.MeshMapperConfig(placements))


def create_mesh_composer(device: ttnn.MeshDevice, placements: Iterable[int]) -> ttnn.CppMeshToTensor:
    placements = [p if p is not None else 0 for p in placements]
    return ttnn.create_mesh_composer(device, ttnn.MeshComposerConfig(placements))


def mesh_mapping_to_placement(
    mapping: Mapping[int, int] | Mapping[int | None, int | None] | None, *, mesh_rank: int
) -> tuple[int | None, ...]:
    placements = [None] * mesh_rank

    if mapping:
        for k, v in mapping.items():
            if k is None or v is None:
                continue
            assert k < mesh_rank, f"mesh mapping keys should be smaller than {mesh_rank}, got {k}"
            placements[k] = v

    return tuple(placements)

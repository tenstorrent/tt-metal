# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


def bf16_tensor(
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
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
        mesh_mapper=mesh_mapper,
    )

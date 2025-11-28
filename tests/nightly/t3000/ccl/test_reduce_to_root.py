# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
def test_reduce_to_root_basic(mesh_device):
    # Setup
    root_coord = (1, 0)
    num_devices = 4
    # currently change shape to fit tile, change later to tiny tile, make same shape in terms of number of packets
    l_shape = [8, 128 * 4]  # should be tiny tile (8,256)
    s_shape = [8, 32 * 4]  # should be tiny tile (8,1)
    m_shape = [8, 32 * 4]  # should be tiny tile (8,1)
    intermediate_shapes = [[8, 192 * 4], [2, 8, 32 * 4]]  # should be (8,256) and (2,8,1)  (8, 320) = (8, 256 + 64)
    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    tile = ttnn.Tile((8, 32))
    shard_spec_l_shape = (8, 192)
    shard_spec_sm_shape = (16, 32)
    shard_l_shape = [8, 128]
    shard_s_shape = [8, 32]

    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((4, 1)))

    # Shard spec: all tensors sharded on core (0,0)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))})
    shard_spec_l = ttnn.ShardSpec(
        shard_grid,
        shard_l_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    shard_spec_s = ttnn.ShardSpec(
        shard_grid,
        shard_s_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    shard_spec_int_l = ttnn.ShardSpec(
        shard_grid,
        shard_spec_l_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    shard_spec_sm = ttnn.ShardSpec(
        shard_grid,
        shard_spec_sm_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config_l = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_l
    )
    mem_config_s = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_s
    )

    mesh_config_int_l = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_int_l
    )

    mem_config_sm = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_sm
    )

    mesh_mapper_config = ttnn.MeshMapperConfig(
        [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], submesh_device.shape
    )
    mesh_mapper = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config)

    for i in range(1):
        l_tensor = ttnn.from_torch(
            torch.ones(l_shape, dtype=torch.bfloat16) * (i + 1),
            device=submesh_device,
            layout=layout,
            tile=tile,
            dtype=dtype,
            memory_config=mem_config_l,
            mesh_mapper=mesh_mapper,
        )
        s_tensor = ttnn.from_torch(
            torch.ones(s_shape, dtype=torch.bfloat16) * (i + 1),
            device=submesh_device,
            layout=layout,
            tile=tile,
            dtype=dtype,
            memory_config=mem_config_s,
            mesh_mapper=mesh_mapper,
        )
        m_tensor = ttnn.from_torch(
            torch.ones(m_shape, dtype=torch.bfloat16) * (i + 1),
            device=submesh_device,
            layout=layout,
            tile=tile,
            dtype=dtype,
            memory_config=mem_config_s,
            mesh_mapper=mesh_mapper,
        )

    # Create intermediate tensors
    intermediate_l = ttnn.from_torch(
        torch.zeros(intermediate_shapes[0], dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mesh_config_int_l,
        mesh_mapper=mesh_mapper,
    )
    intermediate_sm = ttnn.from_torch(
        torch.zeros(intermediate_shapes[1], dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_sm,
        mesh_mapper=mesh_mapper,
    )
    print("intermediate tensor sm data: ", intermediate_sm)

    # Run reduce_to_root operation
    out_l, out_s, out_m = ttnn.reduce_to_root(
        l_tensor,
        s_tensor,
        m_tensor,
        root_coord=ttnn.MeshCoordinate(root_coord),
        intermediate_tensor_l=intermediate_l,
        intermediate_tensor_s_m=intermediate_sm,
        topology=ttnn.Topology.Linear,
    )

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
    l_shape = [32, 64]  # should be tiny tile (8,256)
    s_shape = [32, 32]  # should be tiny tile (8,1)
    m_shape = [32, 32]  # should be tiny tile (8,1)
    intermediate_shapes = [[32, 64], [2, 32, 32]]  # should be (8,256) and (2,8,1)
    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT

    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((4, 1)))

    # Shard spec: all tensors sharded on core (0,0)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard_spec_l = ttnn.ShardSpec(
        shard_grid,
        l_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    shard_spec_s = ttnn.ShardSpec(
        shard_grid,
        s_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    shard_spec_sm = ttnn.ShardSpec(
        shard_grid,
        (64, 32),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config_l = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_l
    )
    mem_config_s = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_s
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
            dtype=dtype,
            memory_config=mem_config_l,
            mesh_mapper=mesh_mapper,
        )
        s_tensor = ttnn.from_torch(
            torch.ones(s_shape, dtype=torch.bfloat16) * (i + 1),
            device=submesh_device,
            layout=layout,
            dtype=dtype,
            memory_config=mem_config_s,
            mesh_mapper=mesh_mapper,
        )
        m_tensor = ttnn.from_torch(
            torch.ones(m_shape, dtype=torch.bfloat16) * (i + 1),
            device=submesh_device,
            layout=layout,
            dtype=dtype,
            memory_config=mem_config_s,
            mesh_mapper=mesh_mapper,
        )

    # Create intermediate tensors
    intermediate_l = ttnn.from_torch(
        torch.zeros(intermediate_shapes[0], dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        dtype=dtype,
        memory_config=mem_config_l,
        mesh_mapper=mesh_mapper,
    )
    intermediate_sm = ttnn.from_torch(
        torch.zeros(intermediate_shapes[1], dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
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

    # Convert output to torch for validation
    out_l_torch = ttnn.to_torch(out_l)
    out_s_torch = ttnn.to_torch(out_s)
    out_m_torch = ttnn.to_torch(out_m)

    # Goldens: sum across devices
    l_golden = torch.zeros(l_shape, dtype=torch.bfloat16)
    s_golden = torch.zeros(s_shape, dtype=torch.bfloat16)
    m_golden = torch.zeros(m_shape, dtype=torch.bfloat16)
    for i in range(num_devices):
        l_golden += torch.ones(l_shape, dtype=torch.bfloat16) * (i + 1)
        s_golden += torch.ones(s_shape, dtype=torch.bfloat16) * (i + 1)
        m_golden += torch.ones(m_shape, dtype=torch.bfloat16) * (i + 1)

    # Validate only on root device
    assert torch.allclose(out_l_torch, l_golden)
    assert torch.allclose(out_s_torch, s_golden)
    assert torch.allclose(out_m_torch, m_golden)

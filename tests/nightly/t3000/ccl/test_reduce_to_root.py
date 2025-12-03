# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc


@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}),
    ],
    indirect=["device_params"],
    ids=["fabric_1d_linear"],
)
def test_reduce_to_root_basic(bh_2d_mesh_device):
    # Setup
    num_devices = 4
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    root_coord = (1, 0)
    num_cores = 8

    # currently change shape to fit tile, change later to tiny tile, make same shape in terms of number of packets
    l_shape = [8, 128 * num_cores]
    s_shape = [8, 32 * num_cores]
    m_shape = [8, 32 * num_cores]
    intermediate_shapes = [
        [8, 192 * num_cores],
        [2, 8, 32 * num_cores],
    ]
    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    tile = ttnn.Tile((8, 32))
    shard_spec_l_shape = (8, 192)
    shard_spec_sm_shape = (16, 32)
    shard_l_shape = [8, 128]
    shard_s_shape = [8, 32]

    # Shard spec: all tensors sharded on core (0,0)
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3)),
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 3)),
        }
    )
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

    # Create unique values for each tile for better debugging
    # L tensor: 8×4096 = 16 rows × 128 cols = 16 row tiles × 4 col tiles (each tile is 8×32)
    # Total tiles = 16/1 * 4096/32 = 16 * 128 = 128 tiles per row, but shape is 8 rows
    # Actually: 8/8 rows × 4096/32 cols = 1 × 128 = 128 tiles total
    l_data = torch.zeros(l_shape, dtype=torch.bfloat16)
    for tile_idx in range(64):
        row_start = 0
        col_start = tile_idx * 32
        l_data[row_start : row_start + 8, col_start : col_start + 32] = tile_idx + 1

    # S tensor: 8×256 = 1 row tile × 8 col tiles = 8 tiles total
    s_data = torch.zeros(s_shape, dtype=torch.bfloat16)
    for tile_idx in range(8):
        row_start = 0
        col_start = tile_idx * 32
        s_data[row_start : row_start + 8, col_start : col_start + 32] = 65 + tile_idx

    # M tensor: 8×256 = 1 row tile × 8 col tiles = 8 tiles total
    m_data = torch.zeros(m_shape, dtype=torch.bfloat16)
    for tile_idx in range(8):
        row_start = 0
        col_start = tile_idx * 32
        m_data[row_start : row_start + 8, col_start : col_start + 32] = 73 + tile_idx

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
    print("after creating input tensors")

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

    print("Reduce to root operation completed")

    # Convert mesh tensors to torch using ConcatMeshToTensor (similar to broadcast test)
    # This concatenates all device tensors along the cluster axis (axis 0 for the submesh)
    cluster_axis = 0  # submesh is (num_devices, 1), so cluster axis is 0

    out_l_concat = ttnn.to_torch(
        out_l,
        mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=cluster_axis),
    )
    out_s_concat = ttnn.to_torch(
        out_s,
        mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=cluster_axis),
    )
    out_m_concat = ttnn.to_torch(
        out_m,
        mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=cluster_axis),
    )

    print(f"Concatenated output shapes - L: {out_l_concat.shape}, S: {out_s_concat.shape}, M: {out_m_concat.shape}")

    # Extract the root device portion from the concatenated output
    # root_coord is (1, 0) which means device index 1 in the submesh
    root_device_idx = root_coord[0]

    # For L tensor: shape is [8, 1024], concatenated along dim 0 becomes [32, 1024] for 4 devices
    # Each device slice is [8, 1024], so root device is slice [8:16, :]
    l_slice_size = l_shape[cluster_axis]
    out_l_torch = out_l_concat[root_device_idx * l_slice_size : (root_device_idx + 1) * l_slice_size, :]

    # For S tensor: shape is [8, 256], concatenated along dim 0 becomes [32, 256] for 4 devices
    s_slice_size = s_shape[cluster_axis]
    out_s_torch = out_s_concat[root_device_idx * s_slice_size : (root_device_idx + 1) * s_slice_size, :]

    # For M tensor: shape is [8, 256], concatenated along dim 0 becomes [32, 256] for 4 devices
    m_slice_size = m_shape[cluster_axis]
    out_m_torch = out_m_concat[root_device_idx * m_slice_size : (root_device_idx + 1) * m_slice_size, :]

    print(f"Output L shape (from root device {root_device_idx}): {out_l_torch.shape}")
    print(f"Output S shape (from root device {root_device_idx}): {out_s_torch.shape}")
    print(f"Output M shape (from root device {root_device_idx}): {out_m_torch.shape}")

    print("tensor l: ", out_l_torch)
    print("tensor s: ", out_s_torch)
    print("tensor m: ", out_m_torch)
    eq, output = comp_pcc(out_l_torch, torch.ones_like(out_l_torch) * 0.992, 0.99)
    assert eq, f"FAILED  l tensor: {output}"
    eq, output = comp_pcc(out_s_torch, torch.ones_like(out_s_torch) * 4.0, 0.99)
    assert eq, f"FAILED  s tensor: {output}"
    eq, output = comp_pcc(out_m_torch, torch.ones_like(out_m_torch) * 1.0, 0.99)
    assert eq, f"FAILED  m tensor: {output}"

    print("✓ All output verification checks passed on root device!")
    print(f"  - L tensor: all values are 1.0")
    print(f"  - S tensor: all values are 4.0")
    print(f"  - M tensor: all values are 1.0")

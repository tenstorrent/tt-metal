# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}),
    ],
    indirect=["device_params"],
    ids=["fabric_1d_linear"],
)
def test_simple_reduce_to_root(bh_2d_mesh_device):
    """
    Simple test with predictable values to verify the algorithm.
    All L, S, M values are set to simple constants per device.
    """
    # Setup
    num_devices = 4
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    root_coord = (1, 0)
    root_device_idx = root_coord[0]
    num_cores = 8

    # Tensor shapes
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
    shard_l_shape = [8, 128]
    shard_s_shape = [8, 32]
    shard_spec_l_shape = (8, 192)
    shard_spec_sm_shape = (16, 32)

    # Shard spec
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3)),
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 3)),
        }
    )
    shard_spec_l = ttnn.ShardSpec(shard_grid, shard_l_shape, ttnn.ShardOrientation.ROW_MAJOR)
    shard_spec_s = ttnn.ShardSpec(shard_grid, shard_s_shape, ttnn.ShardOrientation.ROW_MAJOR)
    shard_spec_int_l = ttnn.ShardSpec(shard_grid, shard_spec_l_shape, ttnn.ShardOrientation.ROW_MAJOR)
    shard_spec_sm = ttnn.ShardSpec(shard_grid, shard_spec_sm_shape, ttnn.ShardOrientation.ROW_MAJOR)

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
        [ttnn.PlacementShard(0), ttnn.PlacementReplicate()], submesh_device.shape
    )
    mesh_mapper = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config)

    mesh_mapper_config2 = ttnn.MeshMapperConfig(
        [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], submesh_device.shape
    )
    mesh_mapper2 = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config2)

    # Create simple test data
    # Use simple values: L=1, S=1, M=device_idx
    # This makes it easy to verify manually
    l_data_per_device = []
    s_data_per_device = []
    m_data_per_device = []

    for device_idx in range(num_devices):
        # L: all ones multiplied by (device_idx + 1)
        l_data = torch.ones(l_shape, dtype=torch.bfloat16) * (device_idx + 1)
        # S: all ones
        s_data = torch.ones(s_shape, dtype=torch.bfloat16)
        # M: all same value = device_idx
        m_data = torch.ones(m_shape, dtype=torch.bfloat16) * device_idx

        l_data_per_device.append(l_data)
        s_data_per_device.append(s_data)
        m_data_per_device.append(m_data)

    print("\n=== Input Data ===")
    for idx, (l, s, m) in enumerate(zip(l_data_per_device, s_data_per_device, m_data_per_device)):
        print(f"Device {idx}: L={l[0,0]:.2f}, S={s[0,0]:.2f}, M={m[0,0]:.2f}")

    # Compute reference manually
    # Round 1: device 0 -> device 1
    # L1=1, S1=1, M1=0 from device 0
    # L2=2, S2=1, M2=1 from device 1
    # m_new = max(0, 1) = 1
    # P1 = exp(0-1) = exp(-1) ≈ 0.368
    # P2 = exp(1-1) = exp(0) = 1.0
    # s_new = 1*0.368 + 1*1.0 = 1.368
    # l_new = 1*0.368 + 2*1.0 = 2.368
    print("\n=== Expected Round 1 (device 0->1) ===")
    print(f"M_new = max(0, 1) = 1.0")
    print(f"P1 = exp(0-1) = {torch.exp(torch.tensor(-1.0)):.4f}")
    print(f"P2 = exp(1-1) = {torch.exp(torch.tensor(0.0)):.4f}")
    print(
        f"S_new = 1*{torch.exp(torch.tensor(-1.0)):.4f} + 1*{torch.exp(torch.tensor(0.0)):.4f} = {(1*torch.exp(torch.tensor(-1.0)) + 1*torch.exp(torch.tensor(0.0))):.4f}"
    )
    print(
        f"L_new = 1*{torch.exp(torch.tensor(-1.0)):.4f} + 2*{torch.exp(torch.tensor(0.0)):.4f} = {(1*torch.exp(torch.tensor(-1.0)) + 2*torch.exp(torch.tensor(0.0))):.4f}"
    )

    # Round 1: device 3 -> device 2
    # L1=4, S1=1, M1=3 from device 3
    # L2=3, S2=1, M2=2 from device 2
    # m_new = max(3, 2) = 3
    # P1 = exp(3-3) = 1.0
    # P2 = exp(2-3) = exp(-1) ≈ 0.368
    # s_new = 1*1.0 + 1*0.368 = 1.368
    # l_new = 4*1.0 + 3*0.368 = 5.104
    print("\n=== Expected Round 1 (device 3->2) ===")
    print(f"M_new = max(3, 2) = 3.0")
    print(f"S_new = {(1*torch.exp(torch.tensor(0.0)) + 1*torch.exp(torch.tensor(-1.0))):.4f}")
    print(f"L_new = {(4*torch.exp(torch.tensor(0.0)) + 3*torch.exp(torch.tensor(-1.0))):.4f}")

    # Round 2: result from dev2 -> dev1
    # L1=5.104, S1=1.368, M1=3 from round1(3->2)
    # L2=2.368, S2=1.368, M2=1 from round1(0->1)
    # m_new = max(3, 1) = 3
    # P1 = exp(3-3) = 1.0
    # P2 = exp(1-3) = exp(-2) ≈ 0.135
    # s_new = 1.368*1.0 + 1.368*0.135 = 1.553
    # l_new = 5.104*1.0 + 2.368*0.135 = 5.424
    # l_final = 5.424 / 1.553 = 3.492
    print("\n=== Expected Round 2 (final) ===")
    s_r1_01 = 1 * torch.exp(torch.tensor(-1.0)) + 1 * torch.exp(torch.tensor(0.0))
    l_r1_01 = 1 * torch.exp(torch.tensor(-1.0)) + 2 * torch.exp(torch.tensor(0.0))
    s_r1_32 = 1 * torch.exp(torch.tensor(0.0)) + 1 * torch.exp(torch.tensor(-1.0))
    l_r1_32 = 4 * torch.exp(torch.tensor(0.0)) + 3 * torch.exp(torch.tensor(-1.0))

    p1_final = torch.exp(torch.tensor(3.0 - 3.0))
    p2_final = torch.exp(torch.tensor(1.0 - 3.0))
    s_final = s_r1_32 * p1_final + s_r1_01 * p2_final
    l_inter = l_r1_32 * p1_final + l_r1_01 * p2_final
    l_final = l_inter / s_final

    print(f"M_final = 3.0")
    print(f"S_final = {s_final:.4f}")
    print(f"L_final = {l_final:.4f}")

    # Stack and send to devices
    l_data_all = torch.stack(l_data_per_device, dim=0)
    s_data_all = torch.stack(s_data_per_device, dim=0)
    m_data_all = torch.stack(m_data_per_device, dim=0)

    l_tensor = ttnn.from_torch(
        l_data_all,
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_l,
        mesh_mapper=mesh_mapper,
    )
    s_tensor = ttnn.from_torch(
        s_data_all,
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_s,
        mesh_mapper=mesh_mapper,
    )
    m_tensor = ttnn.from_torch(
        m_data_all,
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_s,
        mesh_mapper=mesh_mapper,
    )

    intermediate_l = ttnn.from_torch(
        torch.zeros(intermediate_shapes[0], dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mesh_config_int_l,
        mesh_mapper=mesh_mapper2,
    )
    intermediate_sm = ttnn.from_torch(
        torch.zeros(intermediate_shapes[1], dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_sm,
        mesh_mapper=mesh_mapper2,
    )

    # Run operation
    out_l, out_s, out_m = ttnn.reduce_to_root(
        l_tensor,
        s_tensor,
        m_tensor,
        root_coord=ttnn.MeshCoordinate(root_coord),
        intermediate_tensor_l=intermediate_l,
        intermediate_tensor_s_m=intermediate_sm,
        topology=ttnn.Topology.Linear,
    )

    # Get results
    out_l_torch = ttnn.to_torch(out_l, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))
    out_s_torch = ttnn.to_torch(out_s, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))
    out_m_torch = ttnn.to_torch(out_m, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))

    out_l_root = out_l_torch[root_device_idx]
    out_s_root = out_s_torch[root_device_idx]
    out_m_root = out_m_torch[root_device_idx]

    print("\n=== Actual Output (Root Device) ===")
    print(f"L[0,0] = {out_l_root[0,0]:.4f} (expected {l_final:.4f})")
    print(f"S[0,0] = {out_s_root[0,0]:.4f} (expected {s_final:.4f})")
    print(f"M[0,0] = {out_m_root[0,0]:.4f} (expected 3.0)")

    # Check results
    assert torch.allclose(out_m_root, torch.ones_like(out_m_root) * 3.0, rtol=0.01, atol=0.01), "M mismatch!"
    assert torch.allclose(out_s_root, torch.ones_like(out_s_root) * s_final, rtol=0.01, atol=0.05), "S mismatch!"
    assert torch.allclose(out_l_root, torch.ones_like(out_l_root) * l_final, rtol=0.01, atol=0.05), "L mismatch!"

    print("\n✅ All outputs match expected values!")

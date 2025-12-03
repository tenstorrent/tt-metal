# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


def compute_reference_reduce_to_root(
    l_data_per_device, s_data_per_device, m_data_per_device, root_device_idx=1, num_cores=8
):
    """
    Compute the reference output for reduce_to_root operation.

    Algorithm (per core):
    Round 1: device 0 -> device 1, device 3 -> device 2
        m_new = max(m1, m2)
        s_new = s1 * exp(m1 - m_new) + s2 * exp(m2 - m_new)
        l_new = l1 * exp(m1 - m_new) + l2 * exp(m2 - m_new)

    Round 2: device 2 -> device 1
        m_new = max(m1, m2)
        s_new = s1 * exp(m1 - m_new) + s2 * exp(m2 - m_new)
        l_new = l1 * exp(m1 - m_new) + l2 * exp(m2 - m_new)
        l_final = l_new / s_new

    Args:
        l_data_per_device: List of L tensors for each device [num_devices, 8, cols]
        s_data_per_device: List of S tensors for each device [num_devices, 8, cols]
        m_data_per_device: List of M tensors for each device [num_devices, 8, cols]
        root_device_idx: Index of the root device (default 1)
        num_cores: Number of cores per device (default 8)

    Returns:
        l_output, s_output, m_output for root device
    """
    num_devices = len(l_data_per_device)

    # Split each device tensor by core (along column dimension)
    # L: [8, 128*num_cores] -> num_cores x [8, 128]
    # S: [8, 32*num_cores] -> num_cores x [8, 32]
    # M: [8, 32*num_cores] -> num_cores x [8, 32]

    def split_by_cores(tensor_list, core_width):
        """Split tensors along column dimension by core"""
        result = []
        for device_tensor in tensor_list:
            cores = torch.chunk(device_tensor, num_cores, dim=1)
            result.append(cores)
        return result  # [num_devices][num_cores][8, core_width]

    l_per_device_per_core = split_by_cores(l_data_per_device, 128)
    s_per_device_per_core = split_by_cores(s_data_per_device, 32)
    m_per_device_per_core = split_by_cores(m_data_per_device, 32)

    # Process each core independently
    l_final_cores = []
    s_final_cores = []
    m_final_cores = []

    for core_idx in range(num_cores):
        # Get data for this core from each device
        l_dev = [l_per_device_per_core[d][core_idx] for d in range(num_devices)]
        s_dev = [s_per_device_per_core[d][core_idx] for d in range(num_devices)]
        m_dev = [m_per_device_per_core[d][core_idx] for d in range(num_devices)]

        # Round 1: Pairwise reduction
        # device 0 -> device 1
        l1_r1, s1_r1, m1_r1 = l_dev[0], s_dev[0], m_dev[0]
        l2_r1, s2_r1, m2_r1 = l_dev[1], s_dev[1], m_dev[1]

        # M and S are [8, 32], L is [8, 128]
        # mul_block_bcast_cols broadcasts column 0 of M to all columns
        # For each row, take column 0 and broadcast to all 128 columns
        m_new_dev1 = torch.maximum(m1_r1, m2_r1)
        # Broadcast column 0 to all columns: [8, 32] -> [8, 128]
        exp_m1_dev1 = torch.exp(m1_r1 - m_new_dev1)[:, :1].expand(-1, 128)  # Take col 0, broadcast to 128 cols
        exp_m2_dev1 = torch.exp(m2_r1 - m_new_dev1)[:, :1].expand(-1, 128)

        s_new_dev1 = s1_r1 * torch.exp(m1_r1 - m_new_dev1) + s2_r1 * torch.exp(m2_r1 - m_new_dev1)
        l_new_dev1 = l1_r1 * exp_m1_dev1 + l2_r1 * exp_m2_dev1

        # device 3 -> device 2
        l1_r2, s1_r2, m1_r2 = l_dev[3], s_dev[3], m_dev[3]
        l2_r2, s2_r2, m2_r2 = l_dev[2], s_dev[2], m_dev[2]

        m_new_dev2 = torch.maximum(m1_r2, m2_r2)
        exp_m1_dev2 = torch.exp(m1_r2 - m_new_dev2)[:, :1].expand(-1, 128)
        exp_m2_dev2 = torch.exp(m2_r2 - m_new_dev2)[:, :1].expand(-1, 128)

        s_new_dev2 = s1_r2 * torch.exp(m1_r2 - m_new_dev2) + s2_r2 * torch.exp(m2_r2 - m_new_dev2)
        l_new_dev2 = l1_r2 * exp_m1_dev2 + l2_r2 * exp_m2_dev2

        # Round 2: device 2 -> device 1 (final reduction)
        l1_final, s1_final, m1_final = l_new_dev2, s_new_dev2, m_new_dev2
        l2_final, s2_final, m2_final = l_new_dev1, s_new_dev1, m_new_dev1

        m_final = torch.maximum(m1_final, m2_final)
        exp_m1_final = torch.exp(m1_final - m_final)[:, :1].expand(-1, 128)
        exp_m2_final = torch.exp(m2_final - m_final)[:, :1].expand(-1, 128)

        s_final = s1_final * torch.exp(m1_final - m_final) + s2_final * torch.exp(m2_final - m_final)
        l_intermediate = l1_final * exp_m1_final + l2_final * exp_m2_final
        l_final = l_intermediate / s_final[:, :1].expand(-1, 128)  # Broadcast s_final column 0 to match L

        l_final_cores.append(l_final)
        s_final_cores.append(s_final)
        m_final_cores.append(m_final)

    # Concatenate results from all cores along column dimension
    l_result = torch.cat(l_final_cores, dim=1)
    s_result = torch.cat(s_final_cores, dim=1)
    m_result = torch.cat(m_final_cores, dim=1)

    return l_result, s_result, m_result


@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}),
    ],
    indirect=["device_params"],
    ids=["fabric_1d_linear"],
)
def test_reduce_to_root(bh_2d_mesh_device):
    # Setup
    num_devices = 4
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    root_coord = (1, 0)
    root_device_idx = root_coord[0]
    num_cores = 8

    # Tensor shapes: per core we have l_shape_per_core, s_shape_per_core, m_shape_per_core
    # Total shape across all cores: multiply by num_cores
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
        [ttnn.PlacementShard(0), ttnn.PlacementReplicate()], submesh_device.shape
    )
    mesh_mapper = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config)

    mesh_mapper_config2 = ttnn.MeshMapperConfig(
        [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], submesh_device.shape
    )
    mesh_mapper2 = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config2)

    # Generate random input data for each device
    torch.manual_seed(42)  # For reproducibility

    # Create different random data for each device
    l_data_per_device = []
    s_data_per_device = []
    m_data_per_device = []

    for device_idx in range(num_devices):
        # Generate random values in a reasonable range
        # Use small random values to avoid numerical overflow in exp()
        # L: logits, small random values
        l_data = torch.randn(l_shape, dtype=torch.bfloat16) * 0.5 + device_idx
        # S: sum of exponentials, should be positive - use small positive values
        s_data = torch.rand(s_shape, dtype=torch.bfloat16) * 0.5 + 1.0 + device_idx * 0.1
        # M: max values - use small range to avoid exp overflow
        m_data = torch.randn(m_shape, dtype=torch.bfloat16) * 0.5 + device_idx

        l_data_per_device.append(l_data)
        s_data_per_device.append(s_data)
        m_data_per_device.append(m_data)

    print("Input data per device (first row, first few columns):")
    for idx, (l, s, m) in enumerate(zip(l_data_per_device, s_data_per_device, m_data_per_device)):
        print(f"  Device {idx}:")
        print(f"    L[0, 0:8] = {l[0, 0:8]}")
        print(f"    S[0, 0:8] = {s[0, 0:8]}")
        print(f"    M[0, 0:8] = {m[0, 0:8]}")
        print(f"    M[0, 0]={m[0,0]:.4f}, M[0,1]={m[0,1]:.4f}, M[0,31]={m[0,31]:.4f}")

    # Compute reference outputs using PyTorch
    l_ref, s_ref, m_ref = compute_reference_reduce_to_root(
        l_data_per_device, s_data_per_device, m_data_per_device, root_device_idx
    )

    print(f"Reference outputs computed:")
    print(f"L shape: {l_ref.shape}, S shape: {s_ref.shape}, M shape: {m_ref.shape}")
    print(f"Reference first element: L[0,0]={l_ref[0,0]:.4f}, S[0,0]={s_ref[0,0]:.4f}, M[0,0]={m_ref[0,0]:.4f}")
    print(f"Reference L[0, 0:8] = {l_ref[0, 0:8]}")
    print(f"Reference S[0, 0:8] = {s_ref[0, 0:8]}")
    print(f"Reference M[0, 0:8] = {m_ref[0, 0:8]}")

    # Create mesh mapper for distributing different data to each device
    # Shard along dimension 0 (first dim) to send different data to each device

    # Stack all device data and send to mesh
    l_data_all = torch.stack(l_data_per_device, dim=0)  # [num_devices, 8, cols]
    s_data_all = torch.stack(s_data_per_device, dim=0)
    m_data_all = torch.stack(m_data_per_device, dim=0)

    print(f"Creating tensors with shapes: L={l_data_all.shape}, S={s_data_all.shape}, M={m_data_all.shape}")

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
    print("Input tensors created on all devices")
    print("their shapes are :", l_tensor.shape, s_tensor.shape, m_tensor.shape)

    # Create intermediate tensors - replicate zeros to all devices
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

    print("reduce_to_root operation completed")

    # Convert outputs back to torch for verification
    # Use ConcatMeshToTensor to collect outputs from all devices
    out_l_torch = ttnn.to_torch(out_l, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))
    out_s_torch = ttnn.to_torch(out_s, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))
    out_m_torch = ttnn.to_torch(out_m, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))

    print(f"\nOutput from ALL devices:")
    for dev_idx in range(num_devices):
        dev_l = out_l_torch[dev_idx]
        dev_s = out_s_torch[dev_idx]
        dev_m = out_m_torch[dev_idx]
        print(f"  Device {dev_idx}: L[0,0]={dev_l[0,0]}, S[0,0]={dev_s[0,0]}, M[0,0]={dev_m[0,0]}")

    # Extract only the root device output (device index 1)
    out_l_root = out_l_torch[root_device_idx]
    out_s_root = out_s_torch[root_device_idx]
    out_m_root = out_m_torch[root_device_idx]

    print(f"Output shapes: L={out_l_root.shape}, S={out_s_root.shape}, M={out_m_root.shape}")
    print(f"Reference shapes: L={l_ref.shape}, S={s_ref.shape}, M={m_ref.shape}")

    print(f"\nActual output (first row):")
    print(f"  L[0, 0:8] = {out_l_root[0, 0:8]}")
    print(f"  S[0, 0:8] = {out_s_root[0, 0:8]}")
    print(f"  M[0, 0:8] = {out_m_root[0, 0:8]}")

    # Compare with reference
    # Use relaxed tolerance for bfloat16 and exponential operations
    # bfloat16 has ~3 decimal digits of precision, and we have multiple operations
    # (exponentials, multiplications, additions, divisions) which accumulate errors
    rtol = 0.02  # 2% relative tolerance for accumulated rounding errors
    atol = 0.1  # 0.1 absolute tolerance (reasonable for values in range 0.5-3.0)

    # Check L tensor
    l_match = torch.allclose(out_l_root, l_ref, rtol=rtol, atol=atol)
    if not l_match:
        l_diff = torch.abs(out_l_root - l_ref)
        l_max_diff = torch.max(l_diff)
        l_mean_diff = torch.mean(l_diff)
        print(f"L tensor mismatch! Max diff: {l_max_diff}, Mean diff: {l_mean_diff}")
        print(f"L output sample:\n{out_l_root[:4, :64]}")
        print(f"L reference sample:\n{l_ref[:4, :64]}")
    else:
        print("✓ L tensor matches reference")

    # Check S tensor - only column 0 is used, other columns can be garbage
    # S is [8, 256] with tiles of (8, 32), so we have 8 tiles
    # Each tile is 8x32, we only care about column 0 of each tile
    # So we check columns: 0, 32, 64, 96, 128, 160, 192, 224 (first col of each tile)
    s_cols_to_check = [i * 32 for i in range(8)]  # [0, 32, 64, 96, 128, 160, 192, 224]
    s_output_col0 = out_s_root[:, s_cols_to_check]
    s_ref_col0 = s_ref[:, s_cols_to_check]

    s_match = torch.allclose(s_output_col0, s_ref_col0, rtol=rtol, atol=atol)
    if not s_match:
        s_diff = torch.abs(s_output_col0 - s_ref_col0)
        s_max_diff = torch.max(s_diff)
        s_mean_diff = torch.mean(s_diff)
        print(f"S tensor (column 0) mismatch! Max diff: {s_max_diff}, Mean diff: {s_mean_diff}")
        print(f"S output (col 0 of each tile):\n{s_output_col0}")
        print(f"S reference (col 0 of each tile):\n{s_ref_col0}")
    else:
        print("✓ S tensor (column 0) matches reference")

    # Check M tensor - also only column 0 is used
    m_cols_to_check = [i * 32 for i in range(8)]  # [0, 32, 64, 96, 128, 160, 192, 224]
    m_output_col0 = out_m_root[:, m_cols_to_check]
    m_ref_col0 = m_ref[:, m_cols_to_check]

    m_match = torch.allclose(m_output_col0, m_ref_col0, rtol=rtol, atol=atol)
    if not m_match:
        m_diff = torch.abs(m_output_col0 - m_ref_col0)
        m_max_diff = torch.max(m_diff)
        m_mean_diff = torch.mean(m_diff)
        print(f"M tensor (column 0) mismatch! Max diff: {m_max_diff}, Mean diff: {m_mean_diff}")
        print(f"M output (col 0 of each tile):\n{m_output_col0}")
        print(f"M reference (col 0 of each tile):\n{m_ref_col0}")
    else:
        print("✓ M tensor (column 0) matches reference")

    # Assert all tensors match
    assert l_match, "L tensor output does not match reference"
    assert s_match, "S tensor output does not match reference"
    assert m_match, "M tensor output does not match reference"

    print("\n✅ All outputs match reference implementation!")

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from models.common.utility_functions import skip_for_wormhole_b0
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test
import math


def compute_reduction(l1, s1, m1, l2, s2, m2, scale_value, l_width=128):
    """
    Compute the online softmax reduction of two partial results.
    Returns (l_new, s_new, m_new)
    """
    m_new = torch.maximum(m1, m2)
    exp_m1 = torch.exp((m1 - m_new) * scale_value)[:, :1].expand(-1, l_width)
    exp_m2 = torch.exp((m2 - m_new) * scale_value)[:, :1].expand(-1, l_width)
    s_new = s1 * torch.exp((m1 - m_new) * scale_value) + s2 * torch.exp((m2 - m_new) * scale_value)
    l_new = l1 * exp_m1 + l2 * exp_m2
    return l_new, s_new, m_new


def compute_reference_reduce_to_all(
    l_data_per_device, s_data_per_device, m_data_per_device, root_device_idx=1, num_cores=8, scale_value=1.0
):
    """
    Compute the reference output for reduce_to_all operation.

    Algorithm:
    Round 1: Neighbor exchange D0<->D1 and D2<->D3
      - All 4 devices compute their local partial reduction
      - D0 and D1 both compute reduction(D0, D1)
      - D2 and D3 both compute reduction(D2, D3)

    Round 2: Neighbor exchange D0<->D3 and D1<->D2
      - All 4 devices exchange their Round 1 results with the other pair
      - All 4 devices compute the final reduction(D0, D1, D2, D3)
    """
    num_devices = len(l_data_per_device)

    def split_by_cores(tensor_list, num_cores):
        result = []
        for device_tensor in tensor_list:
            cores = torch.chunk(device_tensor, num_cores, dim=1)
            result.append(cores)
        return result

    l_per_device_per_core = split_by_cores(l_data_per_device, num_cores)
    s_per_device_per_core = split_by_cores(s_data_per_device, num_cores)
    m_per_device_per_core = split_by_cores(m_data_per_device, num_cores)

    l_final_cores = []
    s_final_cores = []
    m_final_cores = []

    for core_idx in range(num_cores):
        l_dev = [l_per_device_per_core[d][core_idx] for d in range(num_devices)]
        s_dev = [s_per_device_per_core[d][core_idx] for d in range(num_devices)]
        m_dev = [m_per_device_per_core[d][core_idx] for d in range(num_devices)]

        # Round 1: D0<->D1 and D2<->D3 exchanges
        # D0 and D1 both compute reduction(D0, D1)
        l_r1_01, s_r1_01, m_r1_01 = compute_reduction(
            l_dev[0], s_dev[0], m_dev[0], l_dev[1], s_dev[1], m_dev[1], scale_value
        )

        # D2 and D3 both compute reduction(D2, D3)
        l_r1_23, s_r1_23, m_r1_23 = compute_reduction(
            l_dev[2], s_dev[2], m_dev[2], l_dev[3], s_dev[3], m_dev[3], scale_value
        )

        # Round 2: D0<->D3 and D1<->D2 exchanges
        # All devices compute final reduction of (D0+D1) with (D2+D3)
        l_final, s_final, m_final = compute_reduction(l_r1_01, s_r1_01, m_r1_01, l_r1_23, s_r1_23, m_r1_23, scale_value)

        # Final division: l_out = l_final / s_final
        l_out = l_final / s_final[:, :1].expand(-1, l_final.shape[1])

        l_final_cores.append(l_out)
        s_final_cores.append(s_final)
        m_final_cores.append(m_final)

    return torch.cat(l_final_cores, dim=1), torch.cat(s_final_cores, dim=1), torch.cat(m_final_cores, dim=1)


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}),
    ],
    indirect=["device_params"],
    ids=["fabric_1d_ring"],
)
def test_reduce_to_all_simple_ones(bh_2d_mesh_device):
    """Test reduce_to_all operation with all-ones input - simplest case for debugging."""

    print("\n=== Testing reduce_to_all with ALL ONES ===")

    # Setup
    num_devices = 4
    root_coord = (1, 0)
    root_device_idx = root_coord[0]
    num_cores = 8
    l_width = 128  # Width per core for L tensor
    s_m_width = 32  # Width per core for S and M tensors

    batch_size = 8
    l_shape = [batch_size, l_width * num_cores]
    s_shape = [batch_size, s_m_width * num_cores]
    m_shape = [batch_size, s_m_width * num_cores]
    intermediate_shape = [batch_size, 192 * num_cores]

    scale_value = 1.0
    topology = ttnn.Topology.Ring

    # Create submesh device
    validate_test(num_devices, topology, bh_2d_mesh_device.shape, 0)
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))

    # Tensor config
    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    tile = ttnn.Tile((8, 32))

    # mux cores
    mux_cores = [ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 1), ttnn.CoreCoord(2, 2), ttnn.CoreCoord(2, 3)]

    # Shard config
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3)),
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 3)),
        }
    )
    shard_spec_l = ttnn.ShardSpec(shard_grid, [8, 128], ttnn.ShardOrientation.ROW_MAJOR)
    shard_spec_s = ttnn.ShardSpec(shard_grid, [8, 32], ttnn.ShardOrientation.ROW_MAJOR)
    shard_spec_int = ttnn.ShardSpec(shard_grid, [8, 192], ttnn.ShardOrientation.ROW_MAJOR)

    mem_config_l = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_l
    )
    mem_config_s = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_s
    )
    mem_config_int = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_int
    )

    mesh_mapper_config = ttnn.MeshMapperConfig(
        [ttnn.PlacementShard(0), ttnn.PlacementReplicate()], submesh_device.shape
    )
    mesh_mapper = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config)

    mesh_mapper_config2 = ttnn.MeshMapperConfig(
        [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], submesh_device.shape
    )
    mesh_mapper2 = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config2)

    # Create all-ones input tensors for all devices
    l_data_per_device = [torch.ones(l_shape, dtype=torch.bfloat16) for _ in range(num_devices)]
    s_data_per_device = [torch.ones(s_shape, dtype=torch.bfloat16) for _ in range(num_devices)]
    m_data_per_device = [torch.ones(m_shape, dtype=torch.bfloat16) for _ in range(num_devices)]

    # Compute reference
    ref_l, ref_s, ref_m = compute_reference_reduce_to_all(
        l_data_per_device, s_data_per_device, m_data_per_device, root_device_idx, num_cores, scale_value
    )

    # With all ones:
    # Round 1: m_new = max(1,1) = 1, exp(0) = 1, s_new = 1*1 + 1*1 = 2, l_new = 1*1 + 1*1 = 2
    # Round 2: m_new = max(1,1) = 1, exp(0) = 1, s_new = 2*1 + 2*1 = 4, l_new = 2*1 + 2*1 = 4
    # Final: l_out = 4 / 4 = 1
    print(f"Expected result with all-ones: l_final = 4/4 = 1, s_final = 4, m_final = 1")
    print(f"Reference l (first value): {ref_l[0, 0].item():.4f}")
    print(f"Reference s (first value): {ref_s[0, 0].item():.4f}")
    print(f"Reference m (first value): {ref_m[0, 0].item():.4f}")

    # Stack data for mesh tensor
    l_data_all = torch.stack(l_data_per_device, dim=0)
    s_data_all = torch.stack(s_data_per_device, dim=0)
    m_data_all = torch.stack(m_data_per_device, dim=0)

    # Create mesh tensors
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

    # Create intermediate tensors
    fw_intermediate = ttnn.from_torch(
        torch.zeros(intermediate_shape, dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_int,
        mesh_mapper=mesh_mapper2,
    )
    bw_intermediate = ttnn.from_torch(
        torch.zeros(intermediate_shape, dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_int,
        mesh_mapper=mesh_mapper2,
    )
    coord_intermediate = ttnn.from_torch(
        torch.zeros(intermediate_shape, dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_int,
        mesh_mapper=mesh_mapper2,
    )

    # Run reduce_to_all
    print("\nRunning reduce_to_all...")
    output_l, output_s, output_m = ttnn.reduce_to_all(
        l_tensor,
        s_tensor,
        m_tensor,
        root_coord=ttnn.MeshCoordinate(root_coord),
        scale_fp32=scale_value,
        fw_intermediate_tensor=fw_intermediate,
        bw_intermediate_tensor=bw_intermediate,
        coord_intermediate_tensor=coord_intermediate,
        topology=topology,
        input_mux_cores=mux_cores,
    )

    # Synchronize
    ttnn.synchronize_device(submesh_device)
    print("Device synchronized!")

    # Get output from all devices
    output_l_torch = ttnn.to_torch(output_l, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))
    output_s_torch = ttnn.to_torch(output_s, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))
    output_m_torch = ttnn.to_torch(output_m, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))

    # Get root device output
    out_l_root = output_l_torch[root_device_idx]
    out_s_root = output_s_torch[root_device_idx]
    out_m_root = output_m_torch[root_device_idx]

    # Handle both 1D and 2D output tensors
    if out_l_root.dim() == 1:
        print(f"\nDevice output l (first value): {out_l_root[0].item():.4f}")
        print(f"Device output s (first value): {out_s_root[0].item():.4f}")
        print(f"Device output m (first value): {out_m_root[0].item():.4f}")
    else:
        print(f"\nDevice output l (first value): {out_l_root[0, 0].item():.4f}")
        print(f"Device output s (first value): {out_s_root[0, 0].item():.4f}")
        print(f"Device output m (first value): {out_m_root[0, 0].item():.4f}")

    # Check intermediate tensors
    # Intermediate tensor layout per core: [L: 128, S: 32, M: 32] = 192 total
    interm_width_per_core = 192
    print("\n=== Checking intermediate tensors ===")
    coord_interm_torch = ttnn.to_torch(coord_intermediate, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))
    for d in range(num_devices):
        d_slice = coord_interm_torch[d]
        # Handle both 1D and 2D slices
        # For core 0: L at 0, S at 128, M at 160
        if d_slice.dim() == 1:
            l_val = d_slice[0].item()
            s_val = d_slice[l_width].item()
            m_val = d_slice[l_width + s_m_width].item()
        else:
            l_val = d_slice[0, 0].item()
            s_val = d_slice[0, l_width].item()
            m_val = d_slice[0, l_width + s_m_width].item()
        print(f"D{d} coord_intermediate: l={l_val:.4f}, s={s_val:.4f}, m={m_val:.4f}")

    print(f"\nExpected coord_intermediate after Round 1: l=2, s=2, m=1")

    # Check fw_intermediate
    print("\n=== fw_intermediate (Round 1 received from forward neighbor) ===")
    fw_interm_torch = ttnn.to_torch(fw_intermediate, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))
    for d in range(num_devices):
        d_slice = fw_interm_torch[d]
        # Handle both 1D and 2D slices
        if d_slice.dim() == 1:
            l_val = d_slice[0].item()
            s_val = d_slice[l_width].item()
            m_val = d_slice[l_width + s_m_width].item()
        else:
            l_val = d_slice[0, 0].item()
            s_val = d_slice[0, l_width].item()
            m_val = d_slice[0, l_width + s_m_width].item()
        print(f"D{d} fw_intermediate: l={l_val:.4f}, s={s_val:.4f}, m={m_val:.4f}")

    # Check bw_intermediate
    print("\n=== bw_intermediate (Round 2 received from backward neighbor) ===")
    bw_interm_torch = ttnn.to_torch(bw_intermediate, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))
    for d in range(num_devices):
        d_slice = bw_interm_torch[d]
        # Handle both 1D and 2D slices
        if d_slice.dim() == 1:
            l_val = d_slice[0].item()
            s_val = d_slice[l_width].item()
            m_val = d_slice[l_width + s_m_width].item()
        else:
            l_val = d_slice[0, 0].item()
            s_val = d_slice[0, l_width].item()
            m_val = d_slice[0, l_width + s_m_width].item()
        print(f"D{d} bw_intermediate: l={l_val:.4f}, s={s_val:.4f}, m={m_val:.4f}")

    # Check output on all devices
    print("\n=== Output L values on all devices (should all be 1.0) ===")
    for d in range(num_devices):
        d_out = output_l_torch[d]
        if d_out.dim() == 1:
            print(f"D{d} output l (first value): {d_out[0].item():.4f}")
        else:
            print(f"D{d} output l (first value): {d_out[0, 0].item():.4f}")

    # Compare with reference - flatten both to 1D for comparison
    out_flat = out_l_root.flatten().float()
    ref_flat = ref_l.flatten().float()
    max_diff = torch.max(torch.abs(out_flat - ref_flat)).item()
    match = max_diff < 0.05  # Allow some tolerance for bfloat16

    print(f"\nL tensor match: {match}, max_diff: {max_diff:.4f}")

    if match:
        print("\n=== ALL ONES TEST PASSED ===")
    else:
        print("\n=== ALL ONES TEST FAILED ===")

    assert match, f"L tensor mismatch! Max diff: {max_diff}"


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}),
    ],
    indirect=["device_params"],
    ids=["fabric_1d_ring"],
)
def test_reduce_to_all_random(bh_2d_mesh_device):
    """Test reduce_to_all operation with random input values."""

    print("\n=== Testing reduce_to_all with RANDOM values ===")

    # Setup
    num_devices = 4
    root_coord = (1, 0)
    root_device_idx = root_coord[0]
    num_cores = 8
    l_width = 128
    s_m_width = 32

    # Use batch_size=8 to match shard spec height (for tile layout)
    batch_size = 8
    l_shape = [batch_size, l_width * num_cores]
    s_shape = [batch_size, s_m_width * num_cores]
    m_shape = [batch_size, s_m_width * num_cores]
    intermediate_shape = [batch_size, 192 * num_cores]

    scale_value = 1.0
    topology = ttnn.Topology.Ring

    # Create submesh device
    validate_test(num_devices, topology, bh_2d_mesh_device.shape, 0)
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))

    # Tensor config
    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    tile = ttnn.Tile((8, 32))

    # mux cores
    mux_cores = [ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 1), ttnn.CoreCoord(2, 2), ttnn.CoreCoord(2, 3)]

    # Shard config
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3)),
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 3)),
        }
    )
    shard_spec_l = ttnn.ShardSpec(shard_grid, [8, 128], ttnn.ShardOrientation.ROW_MAJOR)
    shard_spec_s = ttnn.ShardSpec(shard_grid, [8, 32], ttnn.ShardOrientation.ROW_MAJOR)
    shard_spec_int = ttnn.ShardSpec(shard_grid, [8, 192], ttnn.ShardOrientation.ROW_MAJOR)

    mem_config_l = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_l
    )
    mem_config_s = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_s
    )
    mem_config_int = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_int
    )

    mesh_mapper_config = ttnn.MeshMapperConfig(
        [ttnn.PlacementShard(0), ttnn.PlacementReplicate()], submesh_device.shape
    )
    mesh_mapper = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config)

    mesh_mapper_config2 = ttnn.MeshMapperConfig(
        [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], submesh_device.shape
    )
    mesh_mapper2 = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config2)

    # Create random input tensors
    torch.manual_seed(42)
    l_data_per_device = [torch.randn(l_shape, dtype=torch.float32).to(torch.bfloat16) for _ in range(num_devices)]
    s_data_per_device = [torch.rand(s_shape, dtype=torch.float32).to(torch.bfloat16) + 0.5 for _ in range(num_devices)]
    m_data_per_device = [torch.randn(m_shape, dtype=torch.float32).to(torch.bfloat16) for _ in range(num_devices)]

    # Compute reference (convert to float32 for accuracy)
    l_data_f32 = [t.float() for t in l_data_per_device]
    s_data_f32 = [t.float() for t in s_data_per_device]
    m_data_f32 = [t.float() for t in m_data_per_device]

    ref_l, ref_s, ref_m = compute_reference_reduce_to_all(
        l_data_f32, s_data_f32, m_data_f32, root_device_idx, num_cores, scale_value
    )
    ref_l = ref_l.to(torch.bfloat16)

    # Stack data for mesh tensor
    l_data_all = torch.stack(l_data_per_device, dim=0)
    s_data_all = torch.stack(s_data_per_device, dim=0)
    m_data_all = torch.stack(m_data_per_device, dim=0)

    # Create mesh tensors
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

    # Create intermediate tensors
    fw_intermediate = ttnn.from_torch(
        torch.zeros(intermediate_shape, dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_int,
        mesh_mapper=mesh_mapper2,
    )
    bw_intermediate = ttnn.from_torch(
        torch.zeros(intermediate_shape, dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_int,
        mesh_mapper=mesh_mapper2,
    )
    coord_intermediate = ttnn.from_torch(
        torch.zeros(intermediate_shape, dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_int,
        mesh_mapper=mesh_mapper2,
    )

    # Run reduce_to_all
    print("Running reduce_to_all (single iteration, no trace)...")
    output_l, output_s, output_m = ttnn.reduce_to_all(
        l_tensor,
        s_tensor,
        m_tensor,
        root_coord=ttnn.MeshCoordinate(root_coord),
        scale_fp32=scale_value,
        fw_intermediate_tensor=fw_intermediate,
        bw_intermediate_tensor=bw_intermediate,
        coord_intermediate_tensor=coord_intermediate,
        topology=topology,
        input_mux_cores=mux_cores,
    )

    # Synchronize
    print("Synchronizing device...")
    ttnn.synchronize_device(submesh_device)
    print("Device synchronized successfully!")

    # Get output from all devices
    # ConcatMeshToTensor(dim=0) gives shape [num_devices, batch_size, width]
    output_l_torch = ttnn.to_torch(output_l, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))
    print(f"Full output L shape: {output_l_torch.shape}")

    # Get root device output - index by device
    out_l_root = output_l_torch[root_device_idx]

    print("\nVerifying output...")
    print(f"Output L shape: {out_l_root.shape}")
    print(f"Reference L shape: {ref_l.shape}")

    # Check intermediate tensors for debugging
    # ConcatMeshToTensor(dim=0) gives shape [num_devices * batch_size, width] for replicated tensors
    # Intermediate tensor layout per core: [L: 128, S: 32, M: 32] = 192 total
    interm_width_per_core = 192
    print("\n=== Checking coord_intermediate tensor (Round 1 results) ===")
    coord_interm_torch = ttnn.to_torch(coord_intermediate, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))
    print(f"coord_interm_torch shape: {coord_interm_torch.shape}")
    for d in range(num_devices):
        d_slice = coord_interm_torch[d]
        print(f"d_slice shape: {d_slice.shape}")
        # Handle both 1D and 2D slices
        # L values for core i are at offset i * 192 (not i * 128)
        if d_slice.dim() == 1:
            l_vals = [d_slice[i * interm_width_per_core].item() for i in range(num_cores)]
        else:
            l_vals = [d_slice[0, i * interm_width_per_core].item() for i in range(num_cores)]
        print(f"D{d} coord_intermediate first L values: {l_vals}")

    # Compute expected Round 1 intermediate values for comparison
    print("\n=== Expected Round 1 intermediate values (D0+D1 for cores 0-3, D2+D3 for all) ===")
    for core_idx in range(num_cores):
        # D0+D1 reduction
        l0 = l_data_f32[0][:, core_idx * l_width : (core_idx + 1) * l_width]
        l1 = l_data_f32[1][:, core_idx * l_width : (core_idx + 1) * l_width]
        s0 = s_data_f32[0][:, core_idx * s_m_width : (core_idx + 1) * s_m_width]
        s1 = s_data_f32[1][:, core_idx * s_m_width : (core_idx + 1) * s_m_width]
        m0 = m_data_f32[0][:, core_idx * s_m_width : (core_idx + 1) * s_m_width]
        m1 = m_data_f32[1][:, core_idx * s_m_width : (core_idx + 1) * s_m_width]

        l_r1_01, s_r1_01, m_r1_01 = compute_reduction(l0, s0, m0, l1, s1, m1, scale_value)

        # D2+D3 reduction
        l2 = l_data_f32[2][:, core_idx * l_width : (core_idx + 1) * l_width]
        l3 = l_data_f32[3][:, core_idx * l_width : (core_idx + 1) * l_width]
        s2 = s_data_f32[2][:, core_idx * s_m_width : (core_idx + 1) * s_m_width]
        s3 = s_data_f32[3][:, core_idx * s_m_width : (core_idx + 1) * s_m_width]
        m2 = m_data_f32[2][:, core_idx * s_m_width : (core_idx + 1) * s_m_width]
        m3 = m_data_f32[3][:, core_idx * s_m_width : (core_idx + 1) * s_m_width]

        l_r1_23, s_r1_23, m_r1_23 = compute_reduction(l2, s2, m2, l3, s3, m3, scale_value)

        print(
            f"Core {core_idx}: expected D0+D1 l={l_r1_01[0, 0].item():.4f}, expected D2+D3 l={l_r1_23[0, 0].item():.4f}"
        )

    # Check fw_intermediate
    print("\n=== fw_intermediate (Round 2 received data) ===")
    fw_interm_torch = ttnn.to_torch(fw_intermediate, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))
    for d in range(num_devices):
        d_slice = fw_interm_torch[d]
        if d_slice.dim() == 1:
            l_vals = [d_slice[i * interm_width_per_core].item() for i in range(num_cores)]
        else:
            l_vals = [d_slice[0, i * interm_width_per_core].item() for i in range(num_cores)]
        print(f"D{d} fw_intermediate first L values: {l_vals}")

    # Check bw_intermediate
    print("\n=== bw_intermediate (Round 2 received data) ===")
    bw_interm_torch = ttnn.to_torch(bw_intermediate, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))
    for d in range(num_devices):
        d_slice = bw_interm_torch[d]
        if d_slice.dim() == 1:
            l_vals = [d_slice[i * interm_width_per_core].item() for i in range(num_cores)]
        else:
            l_vals = [d_slice[0, i * interm_width_per_core].item() for i in range(num_cores)]
        print(f"D{d} bw_intermediate first L values: {l_vals}")

    # Per-core comparison
    print("\n=== Per-core L comparison (first value of each core shard) ===")
    for core_idx in range(num_cores):
        col_start = core_idx * l_width
        # Handle both 1D and 2D output tensors
        if out_l_root.dim() == 1:
            output_val = out_l_root[col_start].item()
        else:
            output_val = out_l_root[0, col_start].item()
        ref_val = ref_l[0, col_start].item()
        diff = abs(output_val - ref_val)
        print(f"Core {core_idx}: output={output_val:.4f}, ref={ref_val:.4f}, diff={diff:.4f}")

    # Overall comparison - flatten both to 1D for comparison
    out_flat = out_l_root.flatten().float()
    ref_flat = ref_l.flatten().float()
    max_diff = torch.max(torch.abs(out_flat - ref_flat)).item()
    match = max_diff < 0.1  # Allow tolerance for bfloat16

    print(f"\nL tensor match: {match}, max_diff: {max_diff:.4f}")

    if not match:
        # Find location of max diff
        diff_tensor = torch.abs(out_flat - ref_flat)
        max_idx = torch.argmax(diff_tensor).item()
        print(f"Max diff at index={max_idx}")
        print(f"Output value: {out_flat[max_idx].item():.4f}")
        print(f"Reference value: {ref_flat[max_idx].item():.4f}")

    assert match, f"L tensor mismatch! Max diff: {max_diff}"


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}),
    ],
    indirect=["device_params"],
    ids=["fabric_1d_ring"],
)
def test_reduce_to_all_distinct_values(bh_2d_mesh_device):
    """
    Test reduce_to_all with distinct values per device to trace neighbor exchange.

    Each device has a distinct constant value:
    - D0: L=10, S=1, M=0
    - D1: L=20, S=1, M=0
    - D2: L=30, S=1, M=0
    - D3: L=40, S=1, M=0

    With S=1 and M=0, the reduction simplifies to:
    - m_new = max(0, 0) = 0
    - exp(0) = 1
    - s_new = s1 + s2 = 2
    - l_new = l1 + l2

    Expected Round 1 results:
    - D0<->D1: l=30, s=2, m=0
    - D2<->D3: l=70, s=2, m=0

    Expected Round 2 results:
    - All devices: l=100, s=4, m=0
    - Final output: l/s = 100/4 = 25

    This test will reveal which devices are actually exchanging data.
    """

    print("\n=== Testing reduce_to_all with DISTINCT values per device ===")
    print("D0=10, D1=20, D2=30, D3=40")
    print("Expected: Round1 D0+D1=30, D2+D3=70 -> Round2 all=100, final=100/4=25")

    # Setup
    num_devices = 4
    root_coord = (1, 0)
    root_device_idx = root_coord[0]
    num_cores = 8
    l_width = 128
    s_m_width = 32

    batch_size = 8
    l_shape = [batch_size, l_width * num_cores]
    s_shape = [batch_size, s_m_width * num_cores]
    m_shape = [batch_size, s_m_width * num_cores]
    intermediate_shape = [batch_size, 192 * num_cores]

    scale_value = 1.0
    topology = ttnn.Topology.Ring

    # Create submesh device
    validate_test(num_devices, topology, bh_2d_mesh_device.shape, 0)
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))

    # Tensor config
    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    tile = ttnn.Tile((8, 32))

    # mux cores
    mux_cores = [ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 1), ttnn.CoreCoord(2, 2), ttnn.CoreCoord(2, 3)]

    # Shard config
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3)),
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 3)),
        }
    )
    shard_spec_l = ttnn.ShardSpec(shard_grid, [8, 128], ttnn.ShardOrientation.ROW_MAJOR)
    shard_spec_s = ttnn.ShardSpec(shard_grid, [8, 32], ttnn.ShardOrientation.ROW_MAJOR)
    shard_spec_int = ttnn.ShardSpec(shard_grid, [8, 192], ttnn.ShardOrientation.ROW_MAJOR)

    mem_config_l = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_l
    )
    mem_config_s = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_s
    )
    mem_config_int = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_int
    )

    mesh_mapper_config = ttnn.MeshMapperConfig(
        [ttnn.PlacementShard(0), ttnn.PlacementReplicate()], submesh_device.shape
    )
    mesh_mapper = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config)

    mesh_mapper_config2 = ttnn.MeshMapperConfig(
        [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], submesh_device.shape
    )
    mesh_mapper2 = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config2)

    # Create distinct input tensors per device
    # D0=10, D1=20, D2=30, D3=40
    device_l_values = [10.0, 20.0, 30.0, 40.0]
    l_data_per_device = [torch.full(l_shape, device_l_values[d], dtype=torch.bfloat16) for d in range(num_devices)]
    s_data_per_device = [torch.ones(s_shape, dtype=torch.bfloat16) for _ in range(num_devices)]
    m_data_per_device = [torch.zeros(m_shape, dtype=torch.bfloat16) for _ in range(num_devices)]

    # Print input values
    print("\n=== Input values ===")
    for d in range(num_devices):
        print(f"D{d}: L={device_l_values[d]}, S=1, M=0")

    # Compute reference
    l_data_f32 = [t.float() for t in l_data_per_device]
    s_data_f32 = [t.float() for t in s_data_per_device]
    m_data_f32 = [t.float() for t in m_data_per_device]

    ref_l, ref_s, ref_m = compute_reference_reduce_to_all(
        l_data_f32, s_data_f32, m_data_f32, root_device_idx, num_cores, scale_value
    )

    print(f"\nReference output L (first value): {ref_l[0, 0].item():.4f}")
    print(f"Expected: (10+20+30+40)/4 = 25")

    # Stack data for mesh tensor
    l_data_all = torch.stack(l_data_per_device, dim=0)
    s_data_all = torch.stack(s_data_per_device, dim=0)
    m_data_all = torch.stack(m_data_per_device, dim=0)

    # Create mesh tensors
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

    # Create intermediate tensors
    fw_intermediate = ttnn.from_torch(
        torch.zeros(intermediate_shape, dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_int,
        mesh_mapper=mesh_mapper2,
    )
    bw_intermediate = ttnn.from_torch(
        torch.zeros(intermediate_shape, dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_int,
        mesh_mapper=mesh_mapper2,
    )
    coord_intermediate = ttnn.from_torch(
        torch.zeros(intermediate_shape, dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_int,
        mesh_mapper=mesh_mapper2,
    )

    # Run reduce_to_all
    print("\nRunning reduce_to_all...")
    output_l, output_s, output_m = ttnn.reduce_to_all(
        l_tensor,
        s_tensor,
        m_tensor,
        root_coord=ttnn.MeshCoordinate(root_coord),
        scale_fp32=scale_value,
        fw_intermediate_tensor=fw_intermediate,
        bw_intermediate_tensor=bw_intermediate,
        coord_intermediate_tensor=coord_intermediate,
        topology=topology,
        input_mux_cores=mux_cores,
    )

    # Synchronize
    ttnn.synchronize_device(submesh_device)
    print("Device synchronized!")

    # Get output from all devices
    output_l_torch = ttnn.to_torch(output_l, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))

    # Analyze intermediate tensors to trace neighbor exchange
    interm_width_per_core = 192

    print("\n" + "=" * 60)
    print("ANALYZING NEIGHBOR EXCHANGE PATTERNS")
    print("=" * 60)

    # Check fw_intermediate - what each device received from forward neighbor in Round 1
    print("\n=== fw_intermediate (what each device RECEIVED from forward neighbor) ===")
    print("Expected for correct Round 1 (D0<->D1, D2<->D3):")
    print("  D0 should receive D1's data (L=20)")
    print("  D1 should receive D0's data (L=10)")
    print("  D2 should receive D3's data (L=40)")
    print("  D3 should receive D2's data (L=30)")
    print("")
    fw_interm_torch = ttnn.to_torch(fw_intermediate, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))
    for d in range(num_devices):
        d_slice = fw_interm_torch[d]
        if d_slice.dim() == 1:
            l_val = d_slice[0].item()
            s_val = d_slice[l_width].item()
            m_val = d_slice[l_width + s_m_width].item()
        else:
            l_val = d_slice[0, 0].item()
            s_val = d_slice[0, l_width].item()
            m_val = d_slice[l_width + s_m_width].item()
        print(f"D{d} fw_intermediate: L={l_val:.1f}, S={s_val:.1f}, M={m_val:.1f}")

        # Identify source based on L value
        source = "?"
        if abs(l_val - 10) < 1:
            source = "D0"
        elif abs(l_val - 20) < 1:
            source = "D1"
        elif abs(l_val - 30) < 1:
            source = "D2"
        elif abs(l_val - 40) < 1:
            source = "D3"
        elif abs(l_val - 30) < 1:
            source = "D0+D1"
        elif abs(l_val - 70) < 1:
            source = "D2+D3"
        print(f"    -> D{d} received from: {source}")

    # Check bw_intermediate - what each device received from backward neighbor
    print("\n=== bw_intermediate (what each device RECEIVED in Round 2) ===")
    print("Expected for correct Round 2 (D0<->D3, D1<->D2):")
    print("  D0 should receive D2+D3 result (L=70, S=2)")
    print("  D1 should receive D2+D3 result (L=70, S=2)")
    print("  D2 should receive D0+D1 result (L=30, S=2)")
    print("  D3 should receive D0+D1 result (L=30, S=2)")
    print("")
    bw_interm_torch = ttnn.to_torch(bw_intermediate, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))
    for d in range(num_devices):
        d_slice = bw_interm_torch[d]
        if d_slice.dim() == 1:
            l_val = d_slice[0].item()
            s_val = d_slice[l_width].item()
            m_val = d_slice[l_width + s_m_width].item()
        else:
            l_val = d_slice[0, 0].item()
            s_val = d_slice[0, l_width].item()
            m_val = d_slice[0, l_width + s_m_width].item()
        print(f"D{d} bw_intermediate: L={l_val:.1f}, S={s_val:.1f}, M={m_val:.1f}")

        # Identify source based on L value
        source = "?"
        if abs(l_val - 10) < 1:
            source = "D0 raw"
        elif abs(l_val - 20) < 1:
            source = "D1 raw"
        elif abs(l_val - 30) < 2:
            source = "D0+D1 or D2 raw"
        elif abs(l_val - 40) < 1:
            source = "D3 raw"
        elif abs(l_val - 70) < 2:
            source = "D2+D3"
        print(f"    -> D{d} received: {source}")

    # Check coord_intermediate - Round 1 computed results (local + neighbor)
    print("\n=== coord_intermediate (Round 1 computed results: local + received) ===")
    print("Expected after Round 1 reduction:")
    print("  D0: reduction(D0, D1) = L=30, S=2, M=0")
    print("  D1: reduction(D1, D0) = L=30, S=2, M=0")
    print("  D2: reduction(D2, D3) = L=70, S=2, M=0")
    print("  D3: reduction(D3, D2) = L=70, S=2, M=0")
    print("")
    coord_interm_torch = ttnn.to_torch(coord_intermediate, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))
    for d in range(num_devices):
        d_slice = coord_interm_torch[d]
        if d_slice.dim() == 1:
            l_val = d_slice[0].item()
            s_val = d_slice[l_width].item()
            m_val = d_slice[l_width + s_m_width].item()
        else:
            l_val = d_slice[0, 0].item()
            s_val = d_slice[0, l_width].item()
            m_val = d_slice[0, l_width + s_m_width].item()
        print(f"D{d} coord_intermediate: L={l_val:.1f}, S={s_val:.1f}, M={m_val:.1f}")

        # Identify what was computed
        computed = "?"
        if abs(l_val - 10) < 1:
            computed = "D0 only (no reduction)"
        elif abs(l_val - 20) < 1:
            computed = "D1 only (no reduction)"
        elif abs(l_val - 30) < 2 and abs(s_val - 2) < 0.5:
            computed = "D0+D1 (correct for D0/D1)"
        elif abs(l_val - 30) < 2 and abs(s_val - 1) < 0.5:
            computed = "D2 only (no reduction)"
        elif abs(l_val - 40) < 1:
            computed = "D3 only (no reduction)"
        elif abs(l_val - 70) < 2 and abs(s_val - 2) < 0.5:
            computed = "D2+D3 (correct for D2/D3)"
        elif abs(l_val - 50) < 2:
            computed = "D0+D3 or D1+D2 (WRONG PAIRS!)"
        elif abs(l_val - 60) < 2:
            computed = "D1+D3 or D0+D2 (WRONG PAIRS!)"
        print(f"    -> D{d} computed: {computed}")

    # Final output
    print("\n=== Final Output L values ===")
    print("Expected: 25.0 on all devices (100/4)")
    for d in range(num_devices):
        d_out = output_l_torch[d]
        if d_out.dim() == 1:
            l_val = d_out[0].item()
        else:
            l_val = d_out[0, 0].item()
        print(f"D{d} output L: {l_val:.4f}")

    # Get root device output for comparison
    out_l_root = output_l_torch[root_device_idx]
    out_flat = out_l_root.flatten().float()
    ref_flat = ref_l.flatten().float()
    max_diff = torch.max(torch.abs(out_flat - ref_flat)).item()
    match = max_diff < 0.5  # Allow some tolerance

    print(f"\nL tensor match: {match}, max_diff: {max_diff:.4f}")

    if match:
        print("\n=== DISTINCT VALUES TEST PASSED ===")
    else:
        print("\n=== DISTINCT VALUES TEST FAILED ===")
        print("Check the intermediate values above to identify neighbor exchange issues.")

    assert match, f"L tensor mismatch! Max diff: {max_diff}"

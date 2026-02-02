# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN PreSDPA Test
Tests pre-SDPA fused operation with CCL Broadcast + RMSNorm + Mcast + Matmul + Gather + RMSNorm2 + Mcast2 + Matmul2
Input, gamma, and output are sharded on a single core per device in the mesh
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.fused_ops.pre_sdpa.op import PreSDPA


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


@pytest.mark.parametrize(
    "mesh_rows, mesh_cols, sender_row, sender_col",
    [
        (4, 2, 1, 0),
    ],
)
@pytest.mark.parametrize("epsilon", [1e-6])
@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("cluster_axis", [0])
@pytest.mark.parametrize("secondary_cluster_axis", [1])
@pytest.mark.parametrize("using_persistent_buffers", [True])
@pytest.mark.parametrize("skip_ccl", [True])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_pre_sdpa_mesh(
    bh_2d_mesh_device,
    mesh_rows,
    mesh_cols,
    sender_row,
    sender_col,
    epsilon,
    use_fp32,
    cluster_axis,
    secondary_cluster_axis,
    using_persistent_buffers,
    skip_ccl,
):
    """Test TTNN pre-SDPA fused operation with CCL broadcast on mesh devices"""

    num_devices = mesh_rows * mesh_cols

    # Validate mesh size
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    # Create submesh used by the test
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))

    # Configure a single worker sub-device covering the full compute grid
    compute_grid_size = submesh.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    submesh.load_sub_device_manager(submesh.create_sub_device_manager([worker_sub_device], 0))
    submesh.set_sub_device_stall_group([ttnn.SubDeviceId(0)])

    # Input tensor shapes
    shape = (1, 7168)
    matmul_weights_shape = (7168, 1536)

    # Matmul2 weights shape: 1536 x (num_cores * 4 * 32)
    matmul2_grid_x = compute_grid_size.x  # 12 for P150, 11 for non-P150
    matmul2_grid_y = 8
    matmul2_num_cores = matmul2_grid_x * matmul2_grid_y  # 96 or 88
    matmul2_width = matmul2_num_cores * 4 * 32  # 12288 or 11264
    matmul2_weights_shape = (1536, matmul2_width)

    # Mcast/gather core coordinates (same as RMSNorm input core)
    mcast_core_x = matmul2_grid_x - 1  # 11 for P150, 10 for non-P150
    mcast_core_y = 9

    tile = ttnn.Tile([1, 32])

    # RMSNorm2 parameters (1536 elements = 3 tiles of 16x32)
    rmsnorm2_width = 1536

    # Create input and gamma PyTorch tensors
    torch.manual_seed(0)
    sender_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(shape, dtype=torch.bfloat16)
    torch_matmul_weights = torch.randn(matmul_weights_shape, dtype=torch.bfloat16)
    torch_rmsnorm2_gamma = torch.randn((1, rmsnorm2_width), dtype=torch.bfloat16)
    torch_matmul2_weights = torch.randn(matmul2_weights_shape, dtype=torch.bfloat16)

    # Shard spec: single core for input, gamma (on mcast/gather core)
    shard_shape = shape
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(mcast_core, mcast_core)}),
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    # Create mesh tensors for input and intermediate (CCL broadcast destination)
    sender_coord = ttnn.MeshCoordinate(sender_row, sender_col)
    device_tensors = []
    intermediate_tensors = []
    for row in range(mesh_rows):
        if skip_ccl:
            # Single-device mode: all devices have the input
            device_tensors.append(sender_input)
        elif row == sender_row:
            device_tensors.append(sender_input)
        else:
            device_tensors.append(torch.zeros_like(sender_input))
        intermediate_tensors.append(torch.zeros_like(sender_input))

    mesh_tensor_torch = torch.cat(device_tensors, dim=0)
    intermediate_mesh_tensor_torch = torch.cat(intermediate_tensors, dim=0)
    mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementReplicate()], submesh.shape)

    input_tensor_mesh = ttnn.from_torch(
        mesh_tensor_torch,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=tile,
        dtype=ttnn.bfloat16,
        memory_config=mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(submesh, mesh_mapper_config),
    )
    intermediate_tensor_mesh = ttnn.from_torch(
        intermediate_mesh_tensor_torch,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=tile,
        dtype=ttnn.bfloat16,
        memory_config=mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(submesh, mesh_mapper_config),
    )

    # Create gamma tensor replicated across mesh
    gamma_tensor = ttnn.from_torch(
        torch_gamma,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=tile,
        dtype=ttnn.bfloat16,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    # Create matmul weights tensor - width sharded on 6x8 grid (48 cores)
    matmul_grid = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 7))
    num_matmul_cores = 6 * 8  # 48 cores
    matmul_shard_shape = (matmul_weights_shape[0], matmul_weights_shape[1] // num_matmul_cores)
    matmul_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({matmul_grid}),
        matmul_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    matmul_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, matmul_shard_spec)

    matmul_weights_tensor = ttnn.from_torch(
        torch_matmul_weights,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=matmul_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    # Create matmul2 weights tensor - width sharded on device grid
    matmul2_grid = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(matmul2_grid_x - 1, matmul2_grid_y - 1))
    matmul2_shard_shape = (matmul2_weights_shape[0], matmul2_weights_shape[1] // matmul2_num_cores)
    matmul2_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({matmul2_grid}),
        matmul2_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    matmul2_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, matmul2_shard_spec
    )

    matmul2_weights_tensor = ttnn.from_torch(
        torch_matmul2_weights,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=matmul2_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    # Create RMSNorm2 gamma tensor sharded on same core
    rmsnorm2_gamma_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(mcast_core, mcast_core)}),
        (1, rmsnorm2_width),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    rmsnorm2_gamma_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, rmsnorm2_gamma_shard_spec
    )
    rmsnorm2_gamma_tensor = ttnn.from_torch(
        torch_rmsnorm2_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=rmsnorm2_gamma_mem_config,
        tile=tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    # Create output tensor - width sharded on same grid as matmul2
    output_shape = (1, matmul2_width)
    output_shard_shape = (1, matmul2_width // matmul2_num_cores)
    output_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({matmul2_grid}),
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    output_tensor = ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    # Create semaphores for CCL broadcast
    num_cores = compute_grid_size.x * compute_grid_size.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, row_wise=True)

    out_ready_semaphore = ttnn.create_global_semaphore(submesh, available_cores, 0)
    barrier_semaphore = ttnn.create_global_semaphore(submesh, available_cores, 0)
    secondary_sync_semaphore = ttnn.create_global_semaphore(submesh, available_cores, 0)
    semaphores = [out_ready_semaphore, barrier_semaphore, secondary_sync_semaphore]

    # Compute expected result using golden function (uses sender's input)
    torch_expected = PreSDPA.golden(
        sender_input,
        torch_gamma,
        torch_matmul_weights,
        torch_rmsnorm2_gamma,
        torch_matmul2_weights,
        epsilon=epsilon,
    )

    logger.info(
        f"Running pre-SDPA with CCL broadcast: sender=({sender_row},{sender_col}), mesh={mesh_rows}x{mesh_cols}"
    )

    # Debug: Verify input tensor distribution before operation
    logger.info("=== DEBUG: Input tensor distribution before operation ===")
    input_torch_before = ttnn.to_torch(input_tensor_mesh, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))
    for device_idx in range(mesh_rows * mesh_cols):
        start = device_idx * shape[0]
        end = start + shape[0]
        device_slice = input_torch_before[start:end, :]
        slice_sum = torch.sum(torch.abs(device_slice)).item()
        slice_max = torch.max(torch.abs(device_slice)).item()
        logger.info(
            f"  Device {device_idx} input: sum={slice_sum:.4f}, max={slice_max:.4f}, expected_nonzero={device_idx // mesh_cols == sender_row}"
        )

    # Debug: Verify sender input matches expected
    sender_device_idx = sender_row * mesh_cols + sender_col
    sender_slice_start = sender_device_idx * shape[0]
    sender_slice_end = sender_slice_start + shape[0]
    sender_slice = input_torch_before[sender_slice_start:sender_slice_end, :]
    passing_input, pcc_input = comp_pcc(sender_input, sender_slice, 0.999)
    logger.info(f"  Sender device {sender_device_idx} input vs expected: {pcc_input}")

    # Run pre-SDPA operation
    result = PreSDPA.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        gamma_tensor,
        matmul_weights_tensor,
        rmsnorm2_gamma_tensor,
        matmul2_weights_tensor,
        output_tensor,
        sender_coord,
        semaphores=semaphores,
        cluster_axis=cluster_axis,
        secondary_cluster_axis=secondary_cluster_axis,
        using_persistent_buffers=using_persistent_buffers,
        epsilon=epsilon,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=skip_ccl,
    )

    ttnn.synchronize_device(submesh)

    # Debug: Check what generic_op returns
    logger.info(f"=== DEBUG: Result type and info ===")
    logger.info(f"  result type: {type(result)}")
    logger.info(f"  output_tensor type: {type(output_tensor)}")

    # Try reading from output_tensor directly instead of result
    logger.info("=== DEBUG: Reading from output_tensor directly ===")
    output_direct_torch = ttnn.to_torch(output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))
    logger.info(f"  output_direct shape: {output_direct_torch.shape}")
    logger.info(f"  output_direct sum: {torch.sum(torch.abs(output_direct_torch)).item():.4f}")
    logger.info(f"  output_direct max: {torch.max(torch.abs(output_direct_torch)).item():.4f}")

    # Also read from result
    logger.info("=== DEBUG: Reading from result ===")

    # Verify output - every device slice should equal the expected result
    output_tensor_torch = ttnn.to_torch(result, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))
    logger.info(f"  result shape: {output_tensor_torch.shape}")
    logger.info(f"  result sum: {torch.sum(torch.abs(output_tensor_torch)).item():.4f}")
    logger.info(f"  result max: {torch.max(torch.abs(output_tensor_torch)).item():.4f}")

    # Expected output info
    logger.info(f"  expected shape: {torch_expected.shape}")
    logger.info(f"  expected sum: {torch.sum(torch.abs(torch_expected)).item():.4f}")
    logger.info(f"  expected max: {torch.max(torch.abs(torch_expected)).item():.4f}")

    # Debug: Print per-device output slices
    logger.info("=== DEBUG: Per-device output slices ===")
    slice_size = output_shape[0]
    actual_width = output_tensor_torch.shape[1]
    expected_width = torch_expected.shape[1]
    logger.info(f"  slice_size={slice_size}, actual_width={actual_width}, expected_width={expected_width}")

    for device_idx in range(mesh_rows * mesh_cols):
        start = device_idx * slice_size
        end = start + slice_size
        device_slice = output_tensor_torch[start:end, :]
        slice_sum = torch.sum(torch.abs(device_slice)).item()
        slice_max = torch.max(torch.abs(device_slice)).item()
        # Check first 128 and last 128 elements
        first_128_sum = torch.sum(torch.abs(device_slice[:, :128])).item()
        last_128_sum = torch.sum(torch.abs(device_slice[:, -128:])).item()
        logger.info(
            f"  Device {device_idx}: sum={slice_sum:.4f}, max={slice_max:.4f}, first128={first_128_sum:.4f}, last128={last_128_sum:.4f}"
        )

    for device_idx in range(mesh_rows * mesh_cols):
        start = device_idx * slice_size
        end = start + slice_size
        # Trim to expected width for comparison
        received = output_tensor_torch[start:end, :expected_width]

        if received.shape != torch_expected.shape:
            logger.error(
                f"Shape mismatch at device {device_idx}: got {received.shape}, expected {torch_expected.shape}"
            )
            continue

        max_diff = torch.max(torch.abs(received - torch_expected)).item()
        mean_diff = torch.mean(torch.abs(received - torch_expected)).item()

        logger.info(f"Device {device_idx}: Max diff={max_diff}, Mean diff={mean_diff}")

        passing, pcc_message = comp_pcc(torch_expected, received, 0.98)
        logger.info(f"Device {device_idx}: {pcc_message}")

        assert passing, f"Device {device_idx} failed: {pcc_message}"

    logger.info("✓ PreSDPA mesh test passed!")

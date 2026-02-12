# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN Post SDPA Fused Op Test with CCL All-Reduce

Tests the full post_sdpa fused operation with CCL all-reduce which implements:
- ScatterHeads: Scatter [8, 512] from 8 input cores to [1, 512] on 64 cores (8x8)
- Matmul1: [1, 512] x [512, 128] -> [1, 128] per core on 64 cores (8x8)
- Gather1: Collect to [1, 8192] on gather core (12, 9)
- Mcast: Broadcast [1, 8192] to 130 cores (13x10 rectangular grid)
- Matmul2: [1, 8192] x [8192, 64] -> [1, 64] per core on 112 active cores
- Gather2: Collect to [1, 7168] on gather core (12, 9)
- CCL All-Reduce: Exchange [1, 7168] between devices, reduce (local + remote + residual)

The mcast grid (13x10=130 cores) includes 18 inactive cores (row 8 cols 8-12, row 9 cols 0-12)
that receive mcast data but skip matmul2 via is_matmul2_core=false.

CCL All-Reduce uses:
- CCL Receiver = Gather core (12, 9): already has local data after Gather2
- CCL Sender = Adjacent core (11, 9): reads from gather core, sends via fabric

Full operation: [1, 512] @ [512, 8192] @ [8192, 7168] -> [1, 7168] per device,
then all-reduce across devices with optional residual add.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.fused_ops.post_sdpa.op import PostSDPA


def create_fabric_router_config(max_payload_size):
    """Helper to create FabricRouterConfig with custom max payload size."""
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


@pytest.mark.parametrize(
    "num_devices, M, K1, intermediate, K2, output_size, in0_dtype, in1_dtype",
    [
        (2, 1, 512, 8192, 8192, 7168, ttnn.bfloat16, ttnn.bfloat8_b),
    ],
)
@pytest.mark.parametrize("cluster_axis", [0])
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)  # Open full mesh, create submesh
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("fuse_residual_add", [False, True])
def test_post_sdpa(
    mesh_device,
    num_devices,
    M,
    K1,
    intermediate,
    K2,
    output_size,
    in0_dtype,
    in1_dtype,
    cluster_axis,
    fuse_residual_add,
):
    """Test full post_sdpa fused operation with CCL all-reduce"""

    # Validate mesh size
    if mesh_device.shape[0] * mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    # Create submesh - fabric requires opening full system mesh first
    submesh = mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))

    # Set up sub-device
    compute_grid_size = submesh.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = submesh.create_sub_device_manager([worker_sub_device], 0)
    submesh.load_sub_device_manager(sub_device_manager)
    submesh.set_sub_device_stall_group(sub_device_stall_group)

    # Tile dimensions
    a_tile = ttnn.Tile([M, 32])  # 1x32 tiles for input/activation
    b_tile = ttnn.Tile([32, 32])  # 32x32 tiles for weights

    # ========================================================================
    # Grid configuration
    # ========================================================================
    # Scatter input grid: 8 cores (could be any layout, e.g., 2x4, 8x1, etc.)
    # For this test, use 8x1 (8 cores in a row)
    SCATTER_INPUT_GRID_X = 8
    SCATTER_INPUT_GRID_Y = 1
    num_scatter_input_cores = SCATTER_INPUT_GRID_X * SCATTER_INPUT_GRID_Y  # 8
    scatter_rows_per_core = 8  # Each input core has 8 rows

    # Matmul1 / Scatter output grid: 8x8 = 64 cores
    MATMUL1_GRID_X = 8
    MATMUL1_GRID_Y = 8
    num_matmul1_cores = MATMUL1_GRID_X * MATMUL1_GRID_Y  # 64

    # Mcast grid: 13x10 = 130 cores (rectangular for efficient mcast)
    MCAST_GRID_X = 13
    MCAST_GRID_Y = 10
    num_mcast_cores = MCAST_GRID_X * MCAST_GRID_Y  # 130

    # Active Matmul2 cores: 112 (rows 0-7 full 13 cols + row 8 cols 0-7)
    # Non-rectangular grid: 13*8 + 8 = 104 + 8 = 112
    num_matmul2_cores = 112

    # Per-core dimensions
    n1_per_core = intermediate // num_matmul1_cores  # 8192 / 64 = 128
    n2_per_core = output_size // num_matmul2_cores  # 7168 / 112 = 64

    logger.info(f"Testing full post_sdpa fused op with CCL all-reduce:")
    logger.info(
        f"  ScatterHeads: [{scatter_rows_per_core}, {K1}] on {num_scatter_input_cores} cores -> [{M}, {K1}] on {num_matmul1_cores} cores"
    )
    logger.info(f"  Matmul1: [{M}, {K1}] x [{K1}, {intermediate}] on {num_matmul1_cores} cores")
    logger.info(f"  Mcast: [{M}, {intermediate}] to {num_mcast_cores} cores (13x10 grid)")
    logger.info(f"  Matmul2: [{M}, {K2}] x [{K2}, {output_size}] on {num_matmul2_cores} active cores")
    logger.info(f"  CCL All-Reduce: [{M}, {output_size}] across {num_devices} devices")
    logger.info(f"  Output: [{M}, {output_size}] (fuse_residual_add={fuse_residual_add})")

    # Create core grids
    scatter_input_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(SCATTER_INPUT_GRID_X - 1, SCATTER_INPUT_GRID_Y - 1))]
    )
    matmul1_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(MATMUL1_GRID_X - 1, MATMUL1_GRID_Y - 1))]
    )
    # Active matmul2 cores: non-rectangular grid (112 cores)
    # - Rows 0-7: all 13 columns = 104 cores
    # - Row 8: columns 0-7 = 8 cores
    matmul2_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(12, 7)),  # 13x8 = 104 cores
            ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(7, 8)),  # 8x1 = 8 cores
        ]
    )
    gather_core = ttnn.CoreCoord(12, 9)
    gather_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(gather_core, gather_core)])

    # ========================================================================
    # Create PyTorch tensors (per-device)
    # ========================================================================
    torch.manual_seed(0)

    # Weights are shared across all devices (replicated)
    # Weights1: [512, 8192]
    torch_weights1 = torch.randn((K1, intermediate), dtype=torch.bfloat16)

    # Weights2: [8192, 7168]
    torch_weights2 = torch.randn((K2, output_size), dtype=torch.bfloat16)

    # Input per device: [64, 512] total (for scatter to distribute)
    # This will be sharded across 8 input cores, each with [8, 512]
    device_inputs = []
    device_input_for_scatter = []  # For scatter input sharding
    for device_idx in range(num_devices):
        torch_input_single = torch.randn((M, K1), dtype=torch.bfloat16)
        device_inputs.append(torch_input_single)
        # Replicate for scatter input cores: [8, 512] per core × 8 cores = [64, 512] total
        torch_input_scatter = torch_input_single.repeat(num_matmul1_cores, 1)  # [64, 512]
        device_input_for_scatter.append(torch_input_scatter)

    # Residual tensor (optional, shared across devices)
    if fuse_residual_add:
        torch_residual = torch.randn((M, output_size), dtype=torch.bfloat16)
    else:
        torch_residual = None

    # ========================================================================
    # Compute golden reference (full CCL all-reduce)
    # ========================================================================
    torch_expected = PostSDPA.golden(
        [inp.float() for inp in device_inputs],
        torch_weights1.float(),
        torch_weights2.float(),
        torch_residual.float() if torch_residual is not None else None,
    ).bfloat16()
    logger.info(f"Golden output shape: {torch_expected.shape}")

    # ========================================================================
    # Create mesh mapper
    # ========================================================================
    mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementReplicate()], submesh.shape)

    # ========================================================================
    # Create scatter input tensor (height-sharded across 8 input cores)
    # Each input core gets [8, 512] (8 rows to be scattered to 8 different output cores)
    # ========================================================================
    scatter_input_shard_shape = (scatter_rows_per_core, K1)  # [8, 512] per core
    scatter_input_shard_spec = ttnn.ShardSpec(
        scatter_input_grid,
        scatter_input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    scatter_input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, scatter_input_shard_spec
    )

    # Concatenate per-device inputs for mesh tensor
    mesh_input_torch = torch.cat(device_input_for_scatter, dim=0)  # [num_devices * 64, 512]
    ttnn_input = ttnn.from_torch(
        mesh_input_torch,
        device=submesh,
        dtype=in0_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=scatter_input_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.create_mesh_mapper(submesh, mesh_mapper_config),
    )
    logger.info(
        f"Created scatter input tensor: shard {scatter_input_shard_shape} on {num_scatter_input_cores} cores per device"
    )

    # ========================================================================
    # Create matmul1 input tensor = scatter output (height-sharded across 64 matmul1 cores)
    # Each core gets [1, 512]
    # ========================================================================
    scatter_output_shard_shape = (M, K1)  # [1, 512] per core
    scatter_output_shard_spec = ttnn.ShardSpec(
        matmul1_grid,
        scatter_output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    scatter_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, scatter_output_shard_spec
    )

    torch_scatter_output_zeros = torch.zeros((num_matmul1_cores, K1), dtype=torch.bfloat16)
    mesh_scatter_output_torch = torch.cat([torch_scatter_output_zeros] * num_devices, dim=0)
    ttnn_scatter_output = ttnn.from_torch(
        mesh_scatter_output_torch,
        device=submesh,
        dtype=in0_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=scatter_output_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.create_mesh_mapper(submesh, mesh_mapper_config),
    )
    logger.info(
        f"Created scatter output tensor (matmul1 input): shard {scatter_output_shard_shape} on {num_matmul1_cores} cores per device"
    )

    # ========================================================================
    # Create weights1 tensor (width-sharded across matmul1 cores, replicated)
    # Each core gets [512, 128]
    # ========================================================================
    weights1_shard_shape = (K1, n1_per_core)  # [512, 128] per core
    weights1_shard_spec = ttnn.ShardSpec(
        matmul1_grid,
        weights1_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    weights1_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, weights1_shard_spec
    )

    # Get single device for weights (they're replicated, so we just need one)
    single_device = ttnn.get_device_tensors(ttnn_input)[0].device()
    ttnn_weights1 = ttnn.from_torch(
        torch_weights1,
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=single_device,
        memory_config=weights1_mem_config,
        tile=b_tile,
    )
    logger.info(f"Created weights1 tensor: shard {weights1_shard_shape} on {num_matmul1_cores} cores")

    # ========================================================================
    # Create weights2 tensor (width-sharded across 112 active matmul2 cores, replicated)
    # Each core gets [8192, 64]
    # ========================================================================
    weights2_shard_shape = (K2, n2_per_core)  # [8192, 64] per core
    weights2_shard_spec = ttnn.ShardSpec(
        matmul2_grid,  # Non-rectangular grid of 112 active cores
        weights2_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    weights2_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, weights2_shard_spec
    )

    ttnn_weights2 = ttnn.from_torch(
        torch_weights2,
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=single_device,
        memory_config=weights2_mem_config,
        tile=b_tile,
    )
    logger.info(f"Created weights2 tensor: shard {weights2_shard_shape} on {num_matmul2_cores} active cores")

    # ========================================================================
    # Create gather1 output tensor (intermediate [1, 8192] on gather core, replicated)
    # ========================================================================
    gather1_output_shard_shape = (M, intermediate)  # [1, 8192]
    gather1_output_shard_spec = ttnn.ShardSpec(
        gather_core_grid,
        gather1_output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    gather1_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gather1_output_shard_spec
    )

    torch_gather1_zeros = torch.zeros((M, intermediate), dtype=torch.bfloat16)
    ttnn_gather1_output = ttnn.from_torch(
        torch_gather1_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=single_device,
        memory_config=gather1_output_mem_config,
        tile=a_tile,
    )
    logger.info(f"Created gather1 output tensor: {gather1_output_shard_shape} on gather core")

    # ========================================================================
    # Create gather2 output tensor (intermediate [1, 7168] on gather core, per device)
    # This tensor backs CB7 and holds gather2 output for CCL to read
    # ========================================================================
    gather2_output_shard_shape = (M, output_size)  # [1, 7168]
    gather2_output_shard_spec = ttnn.ShardSpec(
        gather_core_grid,
        gather2_output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    gather2_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gather2_output_shard_spec
    )

    torch_gather2_zeros = torch.zeros((M, output_size), dtype=torch.bfloat16)
    mesh_gather2_torch = torch.cat([torch_gather2_zeros] * num_devices, dim=0)
    ttnn_gather2_output = ttnn.from_torch(
        mesh_gather2_torch,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=gather2_output_mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(submesh, mesh_mapper_config),
    )
    logger.info(f"Created gather2 output tensor: {gather2_output_shard_shape} on gather core per device")

    # ========================================================================
    # Create CCL intermediate tensor (1x32 tiles to match gather2 output format)
    # Shape: [1, 7168] = 224 tiles of 1x32
    # ========================================================================
    ccl_intermediate_shape = [M, output_size]  # [1, 7168]
    ccl_intermediate_shard_shape = tuple(ccl_intermediate_shape)
    ccl_intermediate_shard_spec = ttnn.ShardSpec(
        gather_core_grid,
        ccl_intermediate_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    ccl_intermediate_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, ccl_intermediate_shard_spec
    )

    torch_ccl_intermediate = torch.zeros(ccl_intermediate_shape, dtype=torch.bfloat16)
    mesh_ccl_intermediate_torch = torch.cat([torch_ccl_intermediate] * num_devices, dim=0)
    ttnn_ccl_intermediate = ttnn.from_torch(
        mesh_ccl_intermediate_torch,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,  # 1x32 tiles to match gather2 output
        dtype=ttnn.bfloat16,
        memory_config=ccl_intermediate_mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(submesh, mesh_mapper_config),
    )
    logger.info(f"Created CCL intermediate tensor: {ccl_intermediate_shape} on gather core per device")

    # ========================================================================
    # Create final output tensor ([1, 7168] on gather core)
    # ========================================================================
    output_shard_shape = (M, output_size)  # [1, 7168]
    output_shard_spec = ttnn.ShardSpec(
        gather_core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    torch_output_zeros = torch.zeros((M, output_size), dtype=torch.bfloat16)
    mesh_output_torch = torch.cat([torch_output_zeros] * num_devices, dim=0)
    ttnn_output = ttnn.from_torch(
        mesh_output_torch,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=output_mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(submesh, mesh_mapper_config),
    )
    logger.info(f"Created output tensor: {output_shard_shape} on gather core per device")

    # ========================================================================
    # Create residual tensor (optional)
    # ========================================================================
    if fuse_residual_add:
        mesh_residual_torch = torch.cat([torch_residual] * num_devices, dim=0)
        ttnn_residual = ttnn.from_torch(
            mesh_residual_torch,
            device=submesh,
            layout=ttnn.TILE_LAYOUT,
            tile=a_tile,
            dtype=ttnn.bfloat16,
            memory_config=output_mem_config,
            mesh_mapper=ttnn.create_mesh_mapper(submesh, mesh_mapper_config),
        )
        logger.info(f"Created residual tensor: {output_shard_shape} on gather core per device")
    else:
        ttnn_residual = None

    # ========================================================================
    # Create global semaphores for CCL
    # ========================================================================
    num_cores = compute_grid_size.x * compute_grid_size.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, row_wise=True)
    semaphore1 = ttnn.create_global_semaphore(submesh, available_cores, 0)
    semaphore2 = ttnn.create_global_semaphore(submesh, available_cores, 0)
    semaphores = [semaphore1, semaphore2]
    logger.info("Created global semaphores for CCL synchronization")

    # ========================================================================
    # Run fused operation
    # ========================================================================
    logger.info("Running full post_sdpa fused operation with CCL all-reduce...")
    ttnn_result = PostSDPA.op(
        ttnn_input,
        ttnn_weights1,
        ttnn_weights2,
        ttnn_scatter_output,
        ttnn_gather1_output,
        ttnn_gather2_output,
        ttnn_ccl_intermediate,
        ttnn_output,
        semaphores,
        cluster_axis=cluster_axis,
        core_mapping=None,  # Use default mapping
        residual_tensor_mesh=ttnn_residual,
        fp32_dest_acc_en=False,
    )
    ttnn.synchronize_device(submesh)

    # Convert back to torch for comparison
    output_torch = ttnn.to_torch(
        ttnn_result,
        mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0),
    )
    logger.info(f"Output shape: {output_torch.shape}")

    # ========================================================================
    # Verify results using PCC (Pearson Correlation Coefficient)
    # All devices should have the same all-reduced result
    # ========================================================================
    all_passed = True
    pcc_threshold = 0.99  # Require 99% correlation
    for device_idx in range(num_devices):
        received = output_torch[device_idx : device_idx + 1, :]

        # Output should be [1, 7168]
        expected_shape = (M, output_size)
        assert received.shape == expected_shape, f"Expected shape {expected_shape}, got {received.shape}"

        # Compare with all-reduced golden reference using PCC
        # All devices should have the same result after all-reduce
        passing, pcc_message = comp_pcc(torch_expected, received, pcc_threshold)
        if not passing:
            logger.error(f"Device {device_idx}: PCC check FAILED - {pcc_message}")
            logger.error(f"Expected: {torch_expected[:, :5]}")
            logger.error(f"Received: {received[:, :5]}")
            all_passed = False
        else:
            logger.info(f"Device {device_idx}: PASSED - {pcc_message}")

    # Cleanup
    submesh.reset_sub_device_stall_group()
    submesh.clear_loaded_sub_device_manager()

    assert all_passed, "Not all devices have the correct output"
    logger.info("✓ Post SDPA full fused op with CCL all-reduce test passed!")

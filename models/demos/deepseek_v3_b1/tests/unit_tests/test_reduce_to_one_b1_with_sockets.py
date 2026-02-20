# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for Reduce-to-One B1 with socket output.

This test validates the 3-level reduction tree with socket output to D2H receiver,
where the reduce-to-one kernel itself sends data via sockets instead of NOC gather.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_for_wormhole_b0
from models.demos.deepseek_v3_b1.micro_ops.reduce_to_one_b1.op import ReduceToOneB1


def setup_reduce_to_one_with_sockets_test(mesh_device):
    """Setup for reduce-to-one with sockets test."""
    # Log mesh device info
    logger.info(f"mesh_device shape: {mesh_device.shape}")
    logger.info(f"mesh_device num_devices: {mesh_device.get_num_devices()}")

    # Validate mesh has enough devices for 4x2 submesh
    mesh_rows, mesh_cols = mesh_device.shape
    if mesh_rows * mesh_cols < 8:
        pytest.skip(f"Need at least 8 devices, got {mesh_rows * mesh_cols}")

    logger.info(f"Mesh is {mesh_rows}x{mesh_cols} = {mesh_rows * mesh_cols} devices")

    # Setup - create 4x2 submesh
    num_devices = 8
    exit_coord = (0, 1)
    root_coord = (1, 1)

    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    logger.info(f"Created submesh with shape: {submesh_device.shape}")

    assert submesh_device.shape == ttnn.MeshShape((4, 2)), f"Expected 4x2 mesh, got {submesh_device.shape}"

    # Tensor shape: (1, 7168) sharded across 8 cores per device
    tensor_shape = [1, 7168]
    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    tile = ttnn.Tile((1, 32))

    # Get optimal cores for DRAM access (same as working test_reduce_to_one_b1.py)
    compute_cores = submesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_cores = len(compute_cores)
    logger.info(f"Using {num_cores} optimal DRAM cores: {compute_cores[:8]}")

    # Build shard grid from optimal cores (use first 8 cores)
    num_shard_cores = 8
    shard_cores = compute_cores[:num_shard_cores]
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in shard_cores})

    shard_shape = [1, 896]  # 7168 / 8 = 896
    shard_spec = ttnn.ShardSpec(
        shard_grid,
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    mem_config = ttnn.MemoryConfig(ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec)

    # Mesh mapper
    mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], submesh_device.shape)
    mesh_mapper = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config)

    # Create 3 intermediate tensors for 3 reduction rounds
    intermediate_tensors = []
    for _ in range(3):
        intermediate_data = torch.zeros([4, 2] + tensor_shape, dtype=torch.bfloat16)
        intermediate_tensor = ttnn.from_torch(
            intermediate_data,
            device=submesh_device,
            layout=layout,
            tile=tile,
            dtype=dtype,
            memory_config=mem_config,
            mesh_mapper=mesh_mapper,
        )
        intermediate_tensors.append(intermediate_tensor)

    # Create output tensor sharded on a single core (bottom-right of compute grid)
    compute_grid = submesh_device.compute_with_storage_grid_size()
    output_core = ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1)
    logger.info(f"Compute grid: {compute_grid}, output core: {output_core}")
    output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(output_core, output_core)})
    output_shard_shape = tensor_shape  # Full tensor on single core
    output_shard_spec = ttnn.ShardSpec(
        output_shard_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, output_shard_spec
    )

    # Create output tensor (zeros, will be filled by reduce_to_one)
    output_data = torch.zeros([4, 2] + tensor_shape, dtype=torch.bfloat16)
    output_tensor = ttnn.from_torch(
        output_data,
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=output_mem_config,
        mesh_mapper=mesh_mapper,
    )
    logger.info(f"Created output tensor sharded across {num_shard_cores} cores")

    # Generate test data
    data_per_device = []
    torch.manual_seed(42)
    for _ in range(num_devices):
        data_per_device.append(torch.randn(tensor_shape, dtype=torch.bfloat16))

    data_all = torch.stack(data_per_device, dim=0)
    data_all = data_all.reshape(4, 2, *tensor_shape)

    # Create input tensor
    input_tensor = ttnn.from_torch(
        data_all,
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config,
        mesh_mapper=mesh_mapper,
    )

    # Compute reference output
    ref_output = ReduceToOneB1.golden(data_per_device)

    # Create 4 semaphores for reduce_to_one (round1, round2, round3, exit)
    # Get first device from submesh for creating semaphores
    compute_grid = submesh_device.compute_with_storage_grid_size()
    num_cores_total = compute_grid.x * compute_grid.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores_total, compute_grid, row_wise=True)
    semaphores = [ttnn.create_global_semaphore(submesh_device, available_cores, 0) for _ in range(4)]

    # Page size matches the shard width
    page_size_bytes = shard_shape[1] * 2  # bfloat16 = 2 bytes

    return {
        "submesh_device": submesh_device,
        "input_tensor": input_tensor,
        "intermediate_tensors": intermediate_tensors,
        "output_tensor": output_tensor,
        "ref_output": ref_output,
        "root_coord": root_coord,
        "exit_coord": exit_coord,
        "semaphores": semaphores,
        "page_size_bytes": page_size_bytes,
        "shard_cores": shard_cores,
    }


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_2D})],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
def test_reduce_to_one_b1_with_sockets(bh_2d_mesh_device):
    """Test reduce-to-one with integrated D2H socket output."""
    logger.info("=" * 80)
    logger.info("TEST: Reduce-to-One B1 with Integrated D2H Sockets")
    logger.info("=" * 80)

    config = setup_reduce_to_one_with_sockets_test(bh_2d_mesh_device)

    logger.info("Executing reduce-to-one with integrated D2H output...")

    # Execute the operation with integrated D2H socket support
    # The D2H infrastructure is created internally and D2H receiver runs in the same program
    output_tensor, d2h_infra = ReduceToOneB1.op(
        input_tensor_mesh=config["input_tensor"],
        intermediate_tensors=config["intermediate_tensors"],
        output_tensor=config["output_tensor"],
        semaphores=config["semaphores"],
        root_coord=config["root_coord"],
        exit_coord=config["exit_coord"],
        enable_d2h_output=True,  # Enable integrated D2H
    )

    logger.info("Operation completed (D2H receiver is running in same program).")
    logger.info("Reading data from D2H socket...")

    # Get D2H socket from returned infrastructure
    d2h_socket = d2h_infra["d2h_socket"]
    termination_semaphore = d2h_infra["d2h_termination_semaphore"]

    # Read data from D2H socket (8 pages, one from each worker core)
    num_pages = 8
    page_size = config["page_size_bytes"]
    total_bytes = num_pages * page_size
    logger.info(f"Reading {num_pages} pages of {page_size} bytes each ({total_bytes} total bytes)")

    num_elements = total_bytes // 2
    received_tensor_torch = torch.zeros(1, num_elements, dtype=torch.bfloat16)
    received_tensor = ttnn.from_torch(received_tensor_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Set page size and read into tensor
    d2h_socket.set_page_size(total_bytes)
    d2h_socket.read_tensor(received_tensor)

    # Convert back to torch for processing
    received_buffer_torch = ttnn.to_torch(received_tensor)

    logger.info(f"Successfully received {num_pages} pages ({total_bytes} bytes) from D2H socket")

    # Terminate D2H operations
    logger.info("Terminating D2H operations...")
    ttnn.reset_global_semaphore_value(termination_semaphore, 1)
    ttnn.synchronize_device(config["submesh_device"])

    logger.info(f"Received tensor shape: {received_buffer_torch.shape}")

    # Get reference output
    ref_output = config["ref_output"]
    logger.info(f"Reference output shape: {ref_output.shape}")
    logger.info(f"Reference first few values: {ref_output[0, :5]}")
    logger.info(f"Received first few values: {received_buffer_torch[0, :5]}")

    # Verify against reference
    # The socket data should match the reduce-to-one result
    rtol = 0.01
    atol = 0.05
    if torch.allclose(received_buffer_torch, ref_output, rtol=rtol, atol=atol):
        logger.info("✅ Data verification PASSED! Socket output matches reduce-to-one reference.")
    else:
        # Calculate error metrics
        abs_diff = torch.abs(received_buffer_torch - ref_output)
        max_abs_diff = torch.max(abs_diff).item()
        mean_abs_diff = torch.mean(abs_diff).item()

        logger.error(f"❌ Data verification FAILED!")
        logger.error(f"  Max absolute difference: {max_abs_diff}")
        logger.error(f"  Mean absolute difference: {mean_abs_diff}")
        logger.error(f"  Reference range: [{ref_output.min()}, {ref_output.max()}]")
        logger.error(f"  Received range: [{received_buffer_torch.min()}, {received_buffer_torch.max()}]")

        # Find positions with largest errors
        top_errors_idx = torch.argsort(abs_diff.flatten(), descending=True)[:10]
        logger.error(f"  Top 10 error positions:")
        for idx in top_errors_idx:
            pos = idx.item()
            logger.error(
                f"    Position {pos}: expected={ref_output.flatten()[pos]}, "
                f"got={received_buffer_torch.flatten()[pos]}, diff={abs_diff.flatten()[pos]}"
            )

        assert False, "Socket data does not match reduce-to-one reference output"

    logger.info("✓ Reduce-to-One B1 with Integrated D2H Sockets test passed!")

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test reduce-to-one B1 operation with D2D_0 → D2D_1 → D2H chain using SocketInterface.

This test validates the complete data path:
  8 Workers → D2D_0 (aggregator) → D2D_1 (forward via SocketInterface) → D2H (host)

Pipeline:
- D2D_0: Integrated into reduce_to_one_b1 (uses d2d_exchange_multiple_senders.cpp)
- D2D_1 + D2H: Managed from test side using SocketInterface
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_for_wormhole_b0
from models.demos.deepseek_v3_b1.micro_ops.d2d_exchange.op import SocketInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface
from models.demos.deepseek_v3_b1.micro_ops.reduce_to_one_b1.op import ReduceToOneB1


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_2D})],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    indirect=True,
)
@pytest.mark.parametrize(
    "tensor_shape",
    [
        [1, 7168],
    ],
)
def test_reduce_to_one_b1_with_d2d1_d2h(
    mesh_device,
    tensor_shape,
):
    logger.info(f"mesh_device shape: {mesh_device.shape}")
    logger.info(f"mesh_device num_devices: {mesh_device.get_num_devices()}")

    # Validate mesh
    mesh_rows, mesh_cols = mesh_device.shape
    if mesh_rows * mesh_cols < 8:
        pytest.skip(f"Need at least 8 devices, got {mesh_rows * mesh_cols}")

    # Create 4x2 submesh
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    logger.info(f"Created submesh with shape: {submesh_device.shape}")
    assert submesh_device.shape == ttnn.MeshShape((4, 2)), f"Expected 4x2 mesh, got {submesh_device.shape}"

    ttnn.enable_asynchronous_slow_dispatch(submesh_device)

    # Configuration
    root_coord = ttnn.MeshCoordinate(1, 1)
    exit_coord = ttnn.MeshCoordinate(0, 1)

    element_size = 2  # bfloat16
    shard_shape = [1, 896]  #
    payload_size_bytes = shard_shape[1] * element_size  # 896 * 2 = 1,792 bytes per worker
    aggregated_size_bytes = 8 * payload_size_bytes  # 8 * 1,792 = 14,336 bytes total

    # Core allocation
    d2d1_receiver_core = ttnn.CoreCoord(1, 1)  # D2D_1 receiver on EXIT/device-1
    d2h_core = ttnn.CoreCoord(2, 2)  # D2H receiver on EXIT/device-1

    d2d1_receiver_mesh_core = ttnn.MeshCoreCoord(exit_coord, d2d1_receiver_core)
    d2h_mesh_core = ttnn.MeshCoreCoord(exit_coord, d2h_core)

    d2h_socket_fifo_size = aggregated_size_bytes * 2
    d2h_socket = ttnn.D2HSocket(submesh_device, d2h_mesh_core, d2h_socket_fifo_size)
    logger.info(f"✓ Created D2H socket EARLY on EXIT device @ {d2h_core} with FIFO size: {d2h_socket_fifo_size} bytes")

    h2d_core = ttnn.CoreCoord(5, 5)  # Dummy H2D on EXIT device
    h2d_mesh_core = ttnn.MeshCoreCoord(exit_coord, h2d_core)
    # Create dummy H2D socket (required by HostInterface but not used in this test)
    h2d_socket = ttnn.H2DSocket(submesh_device, h2d_mesh_core, ttnn.BufferType.L1, 64, ttnn.H2DMode.HOST_PUSH)
    logger.info(f"✓ Created H2D socket EARLY on EXIT device @ {h2d_core}")

    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    tile = ttnn.Tile((1, 32))

    # Get optimal cores for DRAM access
    compute_cores = submesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_cores = len(compute_cores)
    logger.info(f"Using {num_cores} optimal DRAM cores: {compute_cores[:8]}")

    # Build shard grid from optimal cores (use first 8 cores)
    num_shard_cores = 8
    shard_cores = compute_cores[:num_shard_cores]
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in shard_cores})

    shard_shape = [1, 896]  # Each shard: 1 × 896 (28 tiles × 32)
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

    # Create output tensor sharded on a single core
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
    logger.info(f"Created output tensor sharded on single core: {output_core}")

    # Generate test data
    data_per_device = []
    torch.manual_seed(42)
    for device_idx in range(8):
        data = torch.randn(tensor_shape, dtype=torch.bfloat16)
        data_per_device.append(data)

    # Stack and reshape for mesh: [4, 2, 1, 7168]
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
    logger.info(f"Created input tensor with data: {data_per_device[0][0, :5]}")

    # Create synchronization semaphores
    all_devices_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 7))])
    sem_round1 = ttnn.create_global_semaphore(submesh_device, all_devices_grid, 0)
    sem_round2 = ttnn.create_global_semaphore(submesh_device, all_devices_grid, 0)
    sem_round3 = ttnn.create_global_semaphore(submesh_device, all_devices_grid, 0)
    sem_exit = ttnn.create_global_semaphore(submesh_device, all_devices_grid, 0)
    semaphores = [sem_round1, sem_round2, sem_round3, sem_exit]

    logger.info("\n=== Setting up D2D_0 → D2D_1 → D2H chain using SocketInterface ===")

    # Get shard cores for core allocation
    shard_cores = ttnn.corerange_to_cores(shard_grid, row_wise=True)

    # Pipeline (following SocketInterface pattern from test_host_io.py):
    # - D2D_0 aggregator: ROOT1 device @ (12,9) - aggregates from 8 worker cores, sends to D2D_1
    # - D2D_1 sender: ROOT1 device @ (1,1) - receives from D2D_0, sends cross-device
    # - D2D_1 receiver: EXIT device @ (1,1) - receives cross-device, sends to D2H
    # - D2H: EXIT device @ (2,2) - receives from D2D_1 receiver

    # Core coordinates for the D2D_1 SocketInterface
    d2d1_sender_core = ttnn.CoreCoord(1, 1)

    # Mesh core coordinates
    d2d1_sender_mesh_core = ttnn.MeshCoreCoord(root_coord, d2d1_sender_core)

    logger.info(f"Using D2H socket created early with FIFO size: {d2h_socket_fifo_size} bytes")

    # Create HostInterface on EXIT device
    # D2H socket is already created on d2d1_receiver_core (EXIT @ (2,2))
    internal_socket_buffer_size = aggregated_size_bytes
    host_io = HostInterface(
        h2d_socket=h2d_socket,  # Dummy H2D socket (unused)
        d2h_socket=d2h_socket,  # On EXIT @ (2,2)
        h2d_page_size=64,  # Dummy page size for unused H2D
        d2h_page_size=aggregated_size_bytes,
        core_to_core_socket_buffer_size=internal_socket_buffer_size,
        d2h_upstream_core=d2d1_receiver_mesh_core,  # D2D_1 receiver feeds into D2H
    )
    logger.info("Created HostInterface for D2H on EXIT device")

    # Allocate D2D_0's output core
    d2d0_output_core = ttnn.CoreCoord(12, 9)  # Free core on ROOT1 device
    d2d0_output_mesh_core = ttnn.MeshCoreCoord(root_coord, d2d0_output_core)

    socket_interface = SocketInterface(
        aggregated_size_bytes,  # page_size
        aggregated_size_bytes * 2,  # socket_fifo_size
        aggregated_size_bytes,  # data_size_per_transfer
        d2d1_sender_mesh_core,  # send_core_coord
        d2d1_receiver_mesh_core,  # recv_core_coord
        upstream_socket=None,  # Will create from upstream_core_coord
        downstream_socket=host_io.get_upstream_socket(),  # Gets from HostInterface
        upstream_core_coord=d2d0_output_mesh_core,  # D2D_0 output core (sender)
        downstream_core_coord=None,  # Using downstream_socket instead
        mesh_device=submesh_device,
    )
    logger.info("Created SocketInterface for D2D_1 cross-device forwarding (ROOT1 → EXIT)")

    # Create D2D_0 infrastructure
    # Pass the sender socket [0] from SocketInterface's upstream socket pair
    # This is the socket that D2D_0 will write to
    d2d0_downstream_socket = socket_interface.get_upstream_socket()  # Get sender socket [0]
    d2d0_infra = ReduceToOneB1.create_d2d0_infrastructure(
        submesh_device,
        root_coord,
        shard_cores,
        payload_size_bytes,  # Per-worker page size
        d2d0_downstream_socket,
    )
    logger.info(f"Created D2D_0 infrastructure with aggregator at core {d2d0_infra['d2d0_core']}")

    logger.info("\n=== Running Reduce-to-One with integrated D2D_0 ===")

    # Execute reduce-to-one with D2D_0 integration
    # This launches all reduce workers + D2D_0 aggregator
    result = ReduceToOneB1.op(
        input_tensor,
        intermediate_tensors,
        output_tensor,
        semaphores,
        root_coord,
        enable_d2d0_output=True,
        d2d0_infrastructure=d2d0_infra,
    )

    if isinstance(result, tuple):
        output_tensor, d2d_infra = result
    else:
        output_tensor = result

    logger.info("✓ Reduce-to-one with D2D_0 completed")

    logger.info("\n=== Starting HostInterface and D2D_1 pipeline ===")
    host_io.run()
    socket_interface.run()
    logger.info("✓ HostInterface and D2D_1 pipeline started")

    # Read from D2H socket
    logger.info("\n=== Termination Semaphore Set ===")

    termination_semaphore = d2d_infra["d2d0_termination_semaphore"]
    ttnn.reset_global_semaphore_value(termination_semaphore, 1)

    # D2D_1 sends ONE aggregated page containing data from all 8 worker cores
    # Total bytes = 8 workers × 896 elements/worker × 2 bytes/element = 14,336 bytes
    total_bytes = aggregated_size_bytes  # Single aggregated page
    logger.info(f"Reading 1 aggregated page of {total_bytes} bytes")

    num_elements = total_bytes // 2
    received_tensor_torch = torch.zeros(1, num_elements, dtype=torch.bfloat16)
    d2h_output_tensor = ttnn.from_torch(received_tensor_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Set page size and read into tensor
    d2h_socket.set_page_size(total_bytes)
    logger.info("Calling d2h_socket.read_tensor()...")
    d2h_socket.read_tensor(d2h_output_tensor)
    logger.info("✓ Successfully read from D2H socket")

    # Convert to torch
    d2h_result_torch = ttnn.to_torch(d2h_output_tensor)

    logger.info(f"D2H output (first 5): {d2h_result_torch[0, :5]}")

    # Validate D2H output matches expected value
    expected_value = ReduceToOneB1.golden(data_per_device)
    logger.info(f"Expected value (first 5): {expected_value[0, :5]}")
    rtol = 0.01
    atol = 0.05
    assert torch.allclose(
        d2h_result_torch, expected_value, rtol=rtol, atol=atol
    ), f"D2H output mismatch! Expected {expected_value[0, 0]}, got {d2h_result_torch[0, 0]}"
    logger.info("✓ D2H output matches expected value")

    # Terminate socket interface and host interface
    socket_interface.terminate(False)
    host_io.terminate(True)
    ttnn.synchronize_device(submesh_device)
    logger.info("✓ Socket interface and host interface terminated")

    logger.info("\n✓ Test passed: Reduce-to-one with D2D_0 + D2D_1 (SocketInterface) + D2H (HostInterface) successful!")

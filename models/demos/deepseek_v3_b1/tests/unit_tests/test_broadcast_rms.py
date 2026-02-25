# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_slow_dispatch
from models.demos.deepseek_v3_b1.fused_ops.broadcast_rms.op import BroadcastRMSNorm
from models.demos.deepseek_v3_b1.micro_ops.d2d_exchange.op import SocketInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


@pytest.mark.parametrize(
    "mesh_rows, mesh_cols, sender_row, sender_col, output_shape, input_shard_shape, tensor_mem_layout",
    [
        (
            4,
            2,
            1,
            0,
            [1, 7168],
            (1, 7168),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("cluster_axis", [0])
@pytest.mark.parametrize("secondary_cluster_axis", [1])
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("use_socket", [True, False])
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
def test_broadcast_rms_fused(
    bh_2d_mesh_device,
    mesh_rows,
    mesh_cols,
    sender_row,
    sender_col,
    output_shape,
    input_shard_shape,
    tensor_mem_layout,
    layout,
    input_dtype,
    cluster_axis,
    secondary_cluster_axis,
    num_iters,
    use_socket,
):
    num_devices = mesh_rows * mesh_cols

    # Validate mesh size
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    # Create submesh used by the test
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))

    if use_socket:
        if not is_slow_dispatch():
            pytest.skip("Skipping test in fast dispatch mode")

        ttnn.enable_asynchronous_slow_dispatch(submesh)

    # Configure a single worker sub-device covering the full compute grid
    compute_grid_size = submesh.compute_with_storage_grid_size()

    bcast_core = ttnn.CoreCoord(0, 0)
    pipeline_core = ttnn.CoreCoord(0, 1)
    intermed_core_0 = ttnn.CoreCoord(0, 2)
    intermed_core_1 = ttnn.CoreCoord(0, 3)
    d2h_upstream_core = ttnn.CoreCoord(0, 4)

    # Set up sharded memory config (single core shard like test_ccl_broadcast.py)
    input_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(bcast_core, bcast_core)})
    input_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec)
    output_mem_config = input_mem_config

    # Create sender tensor (the data to broadcast)
    sender_tensor = torch.rand(output_shape, dtype=torch.bfloat16)

    sender_coord = ttnn.MeshCoordinate(sender_row, sender_col)
    # Create mesh tensor with sender's tensor at sender_coord, zeros elsewhere (one slice per row)
    device_tensors = []
    intermediate_tensors = []
    for row in range(mesh_rows):
        if row == sender_row:
            device_tensors.append(sender_tensor)
        else:
            device_tensors.append(torch.zeros_like(sender_tensor))
        intermediate_tensors.append(torch.zeros_like(sender_tensor))

    mesh_tensor_torch = torch.cat(device_tensors, dim=0)
    intermediate_mesh_tensor_torch = torch.cat(intermediate_tensors, dim=0)
    mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementReplicate()], submesh.shape)
    input_tensor_mesh = ttnn.from_torch(
        mesh_tensor_torch,
        device=submesh,
        layout=layout,
        tile=ttnn.Tile((1, 32)),
        dtype=input_dtype,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(submesh, mesh_mapper_config),
    )
    intermediate_tensor_mesh = ttnn.from_torch(
        intermediate_mesh_tensor_torch,
        device=submesh,
        layout=layout,
        tile=ttnn.Tile((1, 32)),
        dtype=input_dtype,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(submesh, mesh_mapper_config),
    )

    # Create gamma tensor - replicate same gamma across the mesh
    torch_gamma = torch.randn(tuple(output_shape), dtype=torch.bfloat16)
    gamma_tensor = ttnn.from_torch(
        torch_gamma,
        device=submesh,
        layout=layout,
        tile=ttnn.Tile((1, 32)),
        dtype=input_dtype,
        memory_config=output_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    # Create output tensor replicated to mesh (destination for broadcast+rms)
    output_tensor = ttnn.from_torch(
        torch.zeros(tuple(output_shape), dtype=torch.bfloat16),
        device=submesh,
        layout=layout,
        tile=ttnn.Tile((1, 32)),
        dtype=input_dtype,
        memory_config=output_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    # Semaphores
    num_cores = compute_grid_size.x * compute_grid_size.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, row_wise=True)

    out_ready_semaphore = ttnn.create_global_semaphore(submesh, available_cores, 0)
    barrier_semaphore = ttnn.create_global_semaphore(submesh, available_cores, 0)
    secondary_sync_semaphore = ttnn.create_global_semaphore(submesh, available_cores, 0)
    semaphores = [out_ready_semaphore, barrier_semaphore, secondary_sync_semaphore]

    host_io = None
    recv_socket = None
    h2d_socket = None
    if use_socket:
        element_size = dtype_size(input_dtype)
        socket_page_size = output_shape[0] * output_shape[1] * element_size
        token_page_size = 64

        sender_device_idx = sender_row * mesh_cols + sender_col
        sender_input = ttnn.get_device_tensors(input_tensor_mesh)[sender_device_idx]

        sender_device_coord = ttnn.MeshCoordinate(sender_row, sender_col)
        pipeline_mesh_core = ttnn.MeshCoreCoord(sender_device_coord, pipeline_core)
        intermed_mesh_core_0 = ttnn.MeshCoreCoord(sender_device_coord, intermed_core_0)
        intermed_mesh_core_1 = ttnn.MeshCoreCoord(sender_device_coord, intermed_core_1)
        bcast_mesh_core = ttnn.MeshCoreCoord(sender_device_coord, bcast_core)
        d2h_upstream_mesh_core = ttnn.MeshCoreCoord(sender_device_coord, d2h_upstream_core)

        sender_tensor_4d = sender_tensor.reshape(1, 1, 1, output_shape[1])
        sender_device = sender_input.device()
        embedding_tensor_device = ttnn.from_torch(
            sender_tensor_4d,
            dtype=input_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=sender_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        h2d_socket = ttnn.H2DSocket(
            submesh, pipeline_mesh_core, ttnn.BufferType.L1, token_page_size * 2, ttnn.H2DMode.HOST_PUSH
        )
        d2h_socket = ttnn.D2HSocket(submesh, pipeline_mesh_core, socket_page_size)

        host_io = HostInterface(
            h2d_socket,
            d2h_socket,
            token_page_size,
            socket_page_size,
            core_to_core_socket_buffer_size=socket_page_size,
            h2d_downstream_core=intermed_mesh_core_0,
            d2h_upstream_core=d2h_upstream_mesh_core,
            embedding_tensor=embedding_tensor_device,
            loopback_mode=False,
            embedding_cb_index=4,
        )

        socket_interface_1 = SocketInterface(
            socket_page_size,
            socket_page_size,
            socket_page_size,
            intermed_mesh_core_0,
            intermed_mesh_core_1,
            upstream_socket=host_io.get_downstream_socket(),
            downstream_core_coord=bcast_mesh_core,
            mesh_device=submesh,
        )

        recv_socket = socket_interface_1.get_downstream_socket()
        host_io.run()
        socket_interface_1.run()

    torch_expected = BroadcastRMSNorm.golden(sender_tensor, torch_gamma)

    # Run fused operation
    logger.info(f"Running fused Broadcast+RMSNorm: sender=({sender_row},{sender_col}), mesh={mesh_rows}x{mesh_cols}")
    result = BroadcastRMSNorm.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        gamma_tensor,
        sender_coord,
        output_tensor,
        semaphores,
        cluster_axis=cluster_axis,
        secondary_cluster_axis=secondary_cluster_axis,
        socket=recv_socket if use_socket else None,
    )

    if use_socket:
        token_size_datums = token_page_size // 4
        torch_token = torch.zeros(1, token_size_datums, dtype=torch.uint32)
        torch_token[0, 0] = 0
        token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        h2d_socket.write_tensor(token_tensor)
        host_io.terminate(False)
        socket_interface_1.terminate(True)
    else:
        ttnn.synchronize_device(submesh)

    # Verify output - every device slice should equal the expected RMSNorm of the sender data
    output_tensor_torch = ttnn.to_torch(result, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))

    slice_size = output_shape[0]
    for device_idx in range(mesh_rows * mesh_cols):
        start = device_idx * slice_size
        end = start + slice_size
        received = output_tensor_torch[start:end, :]

        assert received.shape == torch_expected.shape, f"Shape mismatch at device {device_idx}"

        comp_pcc(received, torch_expected)
        max_diff = torch.max(torch.abs(received - torch_expected)).item()
        mean_diff = torch.mean(torch.abs(received - torch_expected)).item()

        logger.info(f"Max absolute difference: {max_diff}")
        logger.info(f"Mean absolute difference: {mean_diff}")

        passing, pcc_message = comp_pcc(torch_expected, received, 0.999)
        logger.info(pcc_message)

        assert passing, pcc_message

    logger.info("Broadcast+RMSNorm fused test passed!")

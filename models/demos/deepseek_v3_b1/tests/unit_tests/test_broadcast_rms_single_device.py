# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Single-device test for fused Broadcast+RMSNorm op with skip_ccl=True.

When skip_ccl=True, the fused op runs only the RMSNorm portion without CCL broadcast,
making it suitable for single-device execution and testing.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_slow_dispatch
from models.demos.deepseek_v3_b1.fused_ops.broadcast_rms.op import BroadcastRMSNorm
from models.demos.deepseek_v3_b1.micro_ops.d2d_exchange.op import MeshWrapper, SocketInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size


@pytest.mark.parametrize(
    "output_shape, input_shard_shape, tensor_mem_layout",
    [
        ([1, 7168], (1, 7168), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("epsilon", [1e-6])
@pytest.mark.parametrize("fp32_dest_acc_en", [False])
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("use_socket", [True, False])
def test_broadcast_rms_single_device(
    mesh_device,
    output_shape,
    input_shard_shape,
    tensor_mem_layout,
    layout,
    input_dtype,
    epsilon,
    fp32_dest_acc_en,
    use_socket,
):
    """
    Test fused Broadcast+RMSNorm op on a single device with skip_ccl=True for Debugging purpose.

    """

    if use_socket:
        if not is_slow_dispatch():
            pytest.skip("Skipping test in fast dispatch mode")

        ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    bcast_core = ttnn.CoreCoord(0, 0)
    pipeline_core = ttnn.CoreCoord(0, 1)
    intermed_core_0 = ttnn.CoreCoord(0, 2)
    intermed_core_1 = ttnn.CoreCoord(0, 3)
    d2h_upstream_core = ttnn.CoreCoord(0, 4)

    # Set up sharded memory config (single core shard)
    input_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(bcast_core, bcast_core)})
    input_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec)
    output_mem_config = input_mem_config

    # Create input tensor
    torch_input = torch.rand(output_shape, dtype=torch.bfloat16)

    # Create gamma tensor
    torch_gamma = torch.randn(tuple(output_shape), dtype=torch.bfloat16)

    # Convert tensors to device using mesh_device fixture
    input_tensor = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=layout,
        tile=ttnn.Tile((1, 32)),
        dtype=input_dtype,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Intermediate tensor (same shape/config as input)
    intermediate_tensor = ttnn.from_torch(
        torch.zeros_like(torch_input),
        device=mesh_device,
        layout=layout,
        tile=ttnn.Tile((1, 32)),
        dtype=input_dtype,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    gamma_tensor = ttnn.from_torch(
        torch_gamma,
        device=mesh_device,
        layout=layout,
        tile=ttnn.Tile((1, 32)),
        dtype=input_dtype,
        memory_config=output_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Create output tensor
    output_tensor = ttnn.from_torch(
        torch.zeros(tuple(output_shape), dtype=torch.bfloat16),
        device=mesh_device,
        layout=layout,
        tile=ttnn.Tile((1, 32)),
        dtype=input_dtype,
        memory_config=output_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Compute expected output using PyTorch reference
    torch_expected = BroadcastRMSNorm.golden(torch_input, torch_gamma, epsilon=epsilon)

    host_io = None
    recv_socket = None
    h2d_socket = None
    if use_socket:
        element_size = dtype_size(input_dtype)
        socket_page_size = output_shape[0] * output_shape[1] * element_size
        token_page_size = 64

        sender_tensor_4d = torch_input.reshape(1, 1, 1, output_shape[1])
        embedding_tensor = ttnn.from_torch(
            sender_tensor_4d,
            dtype=input_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        device_coord = ttnn.MeshCoordinate(0, 0)
        pipeline_mesh_core = ttnn.MeshCoreCoord(device_coord, pipeline_core)
        intermed_mesh_core_0 = ttnn.MeshCoreCoord(device_coord, intermed_core_0)
        intermed_mesh_core_1 = ttnn.MeshCoreCoord(device_coord, intermed_core_1)
        bcast_mesh_core = ttnn.MeshCoreCoord(device_coord, bcast_core)
        d2h_upstream_mesh_core = ttnn.MeshCoreCoord(device_coord, d2h_upstream_core)

        h2d_socket = ttnn.H2DSocket(
            mesh_device, pipeline_mesh_core, ttnn.BufferType.L1, token_page_size * 2, ttnn.H2DMode.HOST_PUSH
        )
        d2h_socket = ttnn.D2HSocket(mesh_device, pipeline_mesh_core, socket_page_size)

        host_io = HostInterface(
            h2d_socket,
            d2h_socket,
            token_page_size,
            socket_page_size,
            core_to_core_socket_buffer_size=socket_page_size,
            h2d_downstream_core=intermed_mesh_core_0,
            d2h_upstream_core=d2h_upstream_mesh_core,
            embedding_tensor=embedding_tensor,
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
            sender_mesh=MeshWrapper(mesh_device),
            receiver_mesh=MeshWrapper(mesh_device),
        )

        recv_socket = socket_interface_1.get_downstream_socket()

        host_io.run()
        socket_interface_1.run()

    # Run fused operation with skip_ccl=True (single-device mode)
    logger.info("Running fused Broadcast+RMSNorm with skip_ccl=True")
    sender_coord = ttnn.MeshCoordinate(0, 0)  # Ignored when skip_ccl=True

    result = BroadcastRMSNorm.op(
        input_tensor,
        intermediate_tensor,
        gamma_tensor,
        sender_coord,
        output_tensor,
        semaphores=None,  # Not needed when skip_ccl=True
        skip_ccl=True,
        epsilon=epsilon,
        fp32_dest_acc_en=fp32_dest_acc_en,
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
        ttnn.synchronize_device(mesh_device)

    # Convert result back to torch
    output_tensor_torch = ttnn.to_torch(result, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # Verify output
    assert (
        output_tensor_torch.shape == torch_expected.shape
    ), f"Shape mismatch: got {output_tensor_torch.shape}, expected {torch_expected.shape}"

    max_diff = torch.max(torch.abs(output_tensor_torch - torch_expected)).item()
    mean_diff = torch.mean(torch.abs(output_tensor_torch - torch_expected)).item()

    logger.info(f"Max absolute difference: {max_diff}")
    logger.info(f"Mean absolute difference: {mean_diff}")

    passing, pcc_message = comp_pcc(torch_expected, output_tensor_torch, 0.999)
    logger.info(pcc_message)
    assert passing, f"PCC check failed: {pcc_message}"

    logger.info("BroadcastRMSNorm single-device test passed!")

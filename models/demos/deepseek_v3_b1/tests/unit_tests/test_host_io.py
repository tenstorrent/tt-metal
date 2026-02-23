# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for HostInterface H2D/D2H socket loopback.

"""
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.micro_ops.d2d_exchange.op import SocketInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size, ttnn_dtype_from_torch_dtype


def create_fabric_router_config(max_payload_size):
    """Helper to create FabricRouterConfig with custom max payload size."""
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


@pytest.mark.parametrize(
    "tensor_size_bytes, fifo_size, num_iterations",
    [
        (64, 128, 512),
        (64, 256, 512),
        (64, 512, 512),
        (64, 1024, 512),
        (512, 1024, 512),
        (1024, 2048, 128),
        (32768, 65536, 128),
    ],
)
@pytest.mark.parametrize(
    "h2d_mode",
    [
        ttnn.H2DMode.HOST_PUSH,
        ttnn.H2DMode.DEVICE_PULL,
    ],
)
def test_host_io_loopback(mesh_device, tensor_size_bytes, fifo_size, num_iterations, h2d_mode):
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    tensor_size_datums = tensor_size_bytes // 4

    device_coord = ttnn.MeshCoordinate(0, 0)
    core_coord = ttnn.CoreCoord(0, 0)
    socket_core = ttnn.MeshCoreCoord(device_coord, core_coord)

    logger.info("Creating and Running Host Interface")
    h2d_socket = ttnn.H2DSocket(mesh_device, socket_core, ttnn.BufferType.L1, fifo_size, h2d_mode)
    d2h_socket = ttnn.D2HSocket(mesh_device, socket_core, fifo_size)
    host_io = HostInterface(
        h2d_socket,
        d2h_socket,
        tensor_size_bytes,
        tensor_size_bytes,
        core_to_core_socket_buffer_size=fifo_size,
        loopback_mode=True,
    )
    host_io.run()

    logger.info(f"Transferring Data Over H <-> D Interface for {num_iterations} iterations")
    logger.info(f"Tensor Size: {tensor_size_bytes} bytes, FIFO Size: {fifo_size} bytes")
    logger.info(f"H2D Mode: {h2d_mode}")
    logger.info(f"FIFO Size: {fifo_size} bytes")
    for i in range(num_iterations):
        torch_input = torch.arange(i * tensor_size_datums, (i + 1) * tensor_size_datums, dtype=torch.int32).reshape(
            1, tensor_size_datums
        )
        input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        torch_output = torch.zeros(1, tensor_size_datums, dtype=torch.int32)
        output_tensor = ttnn.from_torch(torch_output, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

        h2d_socket.write_tensor(input_tensor)
        d2h_socket.read_tensor(output_tensor)

        result_torch = ttnn.to_torch(output_tensor)
        match = torch.equal(torch_input, result_torch)
        assert match, f"H2D → D2H loopback data mismatch!\nExpected: {torch_input}\nGot: {result_torch}"

    host_io.terminate(True)


@pytest.mark.parametrize(
    "h2d_mode",
    [
        ttnn.H2DMode.HOST_PUSH,
        ttnn.H2DMode.DEVICE_PULL,
    ],
)
@pytest.mark.parametrize(
    "vocab_size, embedding_dim",
    [
        (64, 14336),
        (128, 7168),
        (256, 3584),
        (512, 1792),
    ],
)
@pytest.mark.parametrize(
    "token_fifo_size, embedding_fifo_factor",
    [
        (128, 2),
        (256, 4),
        (512, 8),
    ],
)
def test_host_io_loopback_with_embedding(
    mesh_device, h2d_mode, vocab_size, embedding_dim, token_fifo_size, embedding_fifo_factor
):
    """Test H2D/D2H loopback with an embedding tensor loaded to DRAM."""
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    embedding_dtype = torch.bfloat16
    token_dtype = torch.uint32
    token_size_bytes = 64

    embedding_shape = (1, 1, vocab_size, embedding_dim)
    embedding_fifo_size = embedding_dim * dtype_size(embedding_dtype) * embedding_fifo_factor
    token_size_datums = token_size_bytes // dtype_size(token_dtype)
    embedding_size_bytes = embedding_shape[3] * dtype_size(embedding_dtype)

    device_coord = ttnn.MeshCoordinate(0, 0)
    core_coord = ttnn.CoreCoord(0, 0)
    socket_core = ttnn.MeshCoreCoord(device_coord, core_coord)

    # Create embedding tensor and load to DRAM
    logger.info("Creating embedding tensor and loading to DRAM")
    torch_embedding = torch.randn(embedding_shape, dtype=embedding_dtype)
    embedding_tensor = ttnn.from_torch(
        torch_embedding, dtype=ttnn_dtype_from_torch_dtype(embedding_dtype), layout=ttnn.ROW_MAJOR_LAYOUT
    )
    embedding_tensor = ttnn.to_device(embedding_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Embedding tensor loaded to DRAM with shape {embedding_shape}")

    # Create host interface with embedding
    logger.info("Creating and Running Host Interface with embedding")
    h2d_socket = ttnn.H2DSocket(mesh_device, socket_core, ttnn.BufferType.L1, token_fifo_size, h2d_mode)
    d2h_socket = ttnn.D2HSocket(mesh_device, socket_core, embedding_fifo_size)
    host_io = HostInterface(
        h2d_socket,
        d2h_socket,
        token_size_bytes,
        embedding_size_bytes,
        core_to_core_socket_buffer_size=embedding_fifo_size,
        embedding_tensor=embedding_tensor,
        loopback_mode=True,
    )
    host_io.run()

    embedding_row_elems = embedding_shape[3]
    logger.info(f"Testing embedding with vocab size {vocab_size} over H2D → D2H loopback")

    for token_id in range(vocab_size):
        # Write 64-byte packet with token ID as first uint32, rest zeros
        torch_input = torch.zeros(1, token_size_datums, dtype=token_dtype)
        torch_input[0, 0] = token_id
        input_tensor = ttnn.from_torch(
            torch_input, dtype=ttnn_dtype_from_torch_dtype(token_dtype), layout=ttnn.ROW_MAJOR_LAYOUT
        )

        # Read embedding row back over D2H socket
        torch_output = torch.zeros(1, embedding_row_elems, dtype=embedding_dtype)
        output_tensor = ttnn.from_torch(
            torch_output, dtype=ttnn_dtype_from_torch_dtype(embedding_dtype), layout=ttnn.ROW_MAJOR_LAYOUT
        )

        h2d_socket.write_tensor(input_tensor)
        d2h_socket.read_tensor(output_tensor)

        result_torch = ttnn.to_torch(output_tensor).reshape(-1)
        expected = torch_embedding[0, 0, token_id, :].reshape(-1)
        match = torch.equal(expected, result_torch)
        assert match, (
            f"Token {token_id}: D2H output does not match embedding row!\n"
            f"Expected: {expected[:8]}...\nGot: {result_torch[:8]}..."
        )

    logger.info(f"{vocab_size} token lookups verified successfully")
    host_io.terminate(True)


@pytest.mark.parametrize(
    "tensor_size_bytes, fifo_size, num_iterations",
    [
        (64, 128, 512),
        (64, 256, 512),
        (64, 512, 512),
        (64, 1024, 512),
        (512, 1024, 512),
        (1024, 2048, 128),
        (32768, 65536, 128),
    ],
)
@pytest.mark.parametrize(
    "h2d_mode",
    [
        ttnn.H2DMode.HOST_PUSH,
        ttnn.H2DMode.DEVICE_PULL,
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(7168),
        }
    ],
    indirect=True,
)
def test_multi_stage_pipeline_loopback(mesh_device, tensor_size_bytes, fifo_size, num_iterations, h2d_mode):
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    tensor_size_datums = tensor_size_bytes // 4

    start_device_coord = ttnn.MeshCoordinate(0, 0)
    intermed_device_coord = ttnn.MeshCoordinate(1, 0)
    intermed_device_coord_2 = ttnn.MeshCoordinate(2, 0)
    intermed_device_coord_3 = ttnn.MeshCoordinate(3, 0)
    intermed_device_coord_4 = ttnn.MeshCoordinate(4, 0)
    intermed_device_coord_5 = ttnn.MeshCoordinate(5, 0)
    intermed_device_coord_6 = ttnn.MeshCoordinate(6, 0)
    end_device_coord = ttnn.MeshCoordinate(7, 0)

    entry_core_coord = ttnn.CoreCoord(0, 0)
    exit_core_coord = ttnn.CoreCoord(1, 1)

    h2d_core = ttnn.MeshCoreCoord(start_device_coord, entry_core_coord)
    fwd_core_0 = ttnn.MeshCoreCoord(start_device_coord, exit_core_coord)
    fwd_core_1 = ttnn.MeshCoreCoord(intermed_device_coord, entry_core_coord)
    fwd_core_2 = ttnn.MeshCoreCoord(intermed_device_coord, exit_core_coord)
    fwd_core_3 = ttnn.MeshCoreCoord(intermed_device_coord_2, entry_core_coord)
    fwd_core_4 = ttnn.MeshCoreCoord(intermed_device_coord_2, exit_core_coord)
    fwd_core_5 = ttnn.MeshCoreCoord(intermed_device_coord_3, entry_core_coord)
    fwd_core_6 = ttnn.MeshCoreCoord(intermed_device_coord_3, exit_core_coord)
    fwd_core_7 = ttnn.MeshCoreCoord(intermed_device_coord_4, entry_core_coord)
    fwd_core_8 = ttnn.MeshCoreCoord(intermed_device_coord_4, exit_core_coord)
    fwd_core_9 = ttnn.MeshCoreCoord(intermed_device_coord_5, entry_core_coord)
    fwd_core_10 = ttnn.MeshCoreCoord(intermed_device_coord_5, exit_core_coord)
    fwd_core_11 = ttnn.MeshCoreCoord(intermed_device_coord_6, entry_core_coord)
    fwd_core_12 = ttnn.MeshCoreCoord(intermed_device_coord_6, exit_core_coord)
    fwd_core_13 = ttnn.MeshCoreCoord(end_device_coord, entry_core_coord)
    d2h_core = ttnn.MeshCoreCoord(end_device_coord, exit_core_coord)

    logger.info("Creating and Running Host Interface")

    h2d_socket = ttnn.H2DSocket(mesh_device, h2d_core, ttnn.BufferType.L1, fifo_size, h2d_mode)
    d2h_socket = ttnn.D2HSocket(mesh_device, d2h_core, fifo_size)

    host_io = HostInterface(
        h2d_socket,
        d2h_socket,
        tensor_size_bytes,
        tensor_size_bytes,
        core_to_core_socket_buffer_size=fifo_size,
        h2d_downstream_core=fwd_core_0,
        d2h_upstream_core=fwd_core_13,
    )
    socket_interface_1 = SocketInterface(
        tensor_size_bytes,
        fifo_size,
        tensor_size_bytes,
        fwd_core_0,
        fwd_core_1,
        upstream_socket=host_io.get_downstream_socket(),
        downstream_core_coord=fwd_core_2,
        mesh_device=mesh_device,
    )
    socket_interface_2 = SocketInterface(
        tensor_size_bytes,
        fifo_size,
        tensor_size_bytes,
        fwd_core_2,
        fwd_core_3,
        upstream_socket=socket_interface_1.get_downstream_socket(),
        downstream_core_coord=fwd_core_4,
        mesh_device=mesh_device,
    )
    socket_interface_3 = SocketInterface(
        tensor_size_bytes,
        fifo_size,
        tensor_size_bytes,
        fwd_core_4,
        fwd_core_5,
        upstream_socket=socket_interface_2.get_downstream_socket(),
        downstream_core_coord=fwd_core_6,
        mesh_device=mesh_device,
    )
    socket_interface_4 = SocketInterface(
        tensor_size_bytes,
        fifo_size,
        tensor_size_bytes,
        fwd_core_6,
        fwd_core_7,
        upstream_socket=socket_interface_3.get_downstream_socket(),
        downstream_core_coord=fwd_core_8,
        mesh_device=mesh_device,
    )
    socket_interface_5 = SocketInterface(
        tensor_size_bytes,
        fifo_size,
        tensor_size_bytes,
        fwd_core_8,
        fwd_core_9,
        upstream_socket=socket_interface_4.get_downstream_socket(),
        downstream_core_coord=fwd_core_10,
        mesh_device=mesh_device,
    )
    socket_interface_6 = SocketInterface(
        tensor_size_bytes,
        fifo_size,
        tensor_size_bytes,
        fwd_core_10,
        fwd_core_11,
        upstream_socket=socket_interface_5.get_downstream_socket(),
        downstream_core_coord=fwd_core_12,
        mesh_device=mesh_device,
    )
    socket_interface_7 = SocketInterface(
        tensor_size_bytes,
        fifo_size,
        tensor_size_bytes,
        fwd_core_12,
        fwd_core_13,
        upstream_socket=socket_interface_6.get_downstream_socket(),
        downstream_socket=host_io.get_upstream_socket(),
    )
    host_io.run()
    socket_interface_1.run()
    socket_interface_2.run()
    socket_interface_3.run()
    socket_interface_4.run()
    socket_interface_5.run()
    socket_interface_6.run()
    socket_interface_7.run()

    logger.info(f"Transferring Data Over H <-> D Interface for {num_iterations} iterations")
    logger.info(f"Tensor Size: {tensor_size_bytes} bytes, FIFO Size: {fifo_size} bytes")
    logger.info(f"H2D Mode: {h2d_mode}")
    logger.info(f"FIFO Size: {fifo_size} bytes")
    for i in range(num_iterations):
        torch_input = torch.arange(i * tensor_size_datums, (i + 1) * tensor_size_datums, dtype=torch.int32).reshape(
            1, tensor_size_datums
        )
        input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        torch_output = torch.zeros(1, tensor_size_datums, dtype=torch.int32)
        output_tensor = ttnn.from_torch(torch_output, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

        h2d_socket.write_tensor(input_tensor)
        d2h_socket.read_tensor(output_tensor)

        result_torch = ttnn.to_torch(output_tensor)
        match = torch.equal(torch_input, result_torch)
        assert match, f"H2D → D2H loopback data mismatch!\nExpected: {torch_input}\nGot: {result_torch}"

    host_io.terminate(False)
    socket_interface_1.terminate(False)
    socket_interface_2.terminate(False)
    socket_interface_3.terminate(False)
    socket_interface_4.terminate(False)
    socket_interface_5.terminate(False)
    socket_interface_6.terminate(False)
    socket_interface_7.terminate(True)


@pytest.mark.parametrize(
    "h2d_mode",
    [
        ttnn.H2DMode.HOST_PUSH,
        ttnn.H2DMode.DEVICE_PULL,
    ],
)
@pytest.mark.parametrize(
    "vocab_size, embedding_dim",
    [
        (128, 7168),
        (256, 3584),
        (512, 1792),
    ],
)
@pytest.mark.parametrize(
    "token_fifo_size, embedding_fifo_factor",
    [
        (128, 2),
        (256, 4),
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(7168),
        }
    ],
    indirect=True,
)
def test_multi_stage_pipeline_loopback_with_embedding(
    mesh_device, h2d_mode, vocab_size, embedding_dim, token_fifo_size, embedding_fifo_factor
):
    """Test multi-stage pipeline with embedding: H2D receives token, looks up embedding, streams through all devices, D2H sends embedding row back."""
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    embedding_dtype = torch.bfloat16
    token_dtype = torch.uint32
    token_size_bytes = 64

    embedding_shape = (1, 1, vocab_size, embedding_dim)
    embedding_fifo_size = embedding_dim * dtype_size(embedding_dtype) * embedding_fifo_factor
    token_size_datums = token_size_bytes // dtype_size(token_dtype)
    embedding_size_bytes = embedding_shape[3] * dtype_size(embedding_dtype)

    start_device_coord = ttnn.MeshCoordinate(0, 0)
    intermed_device_coord = ttnn.MeshCoordinate(1, 0)
    intermed_device_coord_2 = ttnn.MeshCoordinate(2, 0)
    intermed_device_coord_3 = ttnn.MeshCoordinate(3, 0)
    intermed_device_coord_4 = ttnn.MeshCoordinate(4, 0)
    intermed_device_coord_5 = ttnn.MeshCoordinate(5, 0)
    intermed_device_coord_6 = ttnn.MeshCoordinate(6, 0)
    end_device_coord = ttnn.MeshCoordinate(7, 0)

    entry_core_coord = ttnn.CoreCoord(0, 0)
    exit_core_coord = ttnn.CoreCoord(1, 1)

    h2d_core = ttnn.MeshCoreCoord(start_device_coord, entry_core_coord)
    fwd_core_0 = ttnn.MeshCoreCoord(start_device_coord, exit_core_coord)
    fwd_core_1 = ttnn.MeshCoreCoord(intermed_device_coord, entry_core_coord)
    fwd_core_2 = ttnn.MeshCoreCoord(intermed_device_coord, exit_core_coord)
    fwd_core_3 = ttnn.MeshCoreCoord(intermed_device_coord_2, entry_core_coord)
    fwd_core_4 = ttnn.MeshCoreCoord(intermed_device_coord_2, exit_core_coord)
    fwd_core_5 = ttnn.MeshCoreCoord(intermed_device_coord_3, entry_core_coord)
    fwd_core_6 = ttnn.MeshCoreCoord(intermed_device_coord_3, exit_core_coord)
    fwd_core_7 = ttnn.MeshCoreCoord(intermed_device_coord_4, entry_core_coord)
    fwd_core_8 = ttnn.MeshCoreCoord(intermed_device_coord_4, exit_core_coord)
    fwd_core_9 = ttnn.MeshCoreCoord(intermed_device_coord_5, entry_core_coord)
    fwd_core_10 = ttnn.MeshCoreCoord(intermed_device_coord_5, exit_core_coord)
    fwd_core_11 = ttnn.MeshCoreCoord(intermed_device_coord_6, entry_core_coord)
    fwd_core_12 = ttnn.MeshCoreCoord(intermed_device_coord_6, exit_core_coord)
    fwd_core_13 = ttnn.MeshCoreCoord(end_device_coord, entry_core_coord)
    d2h_core = ttnn.MeshCoreCoord(end_device_coord, exit_core_coord)

    # Create embedding tensor and load to DRAM
    logger.info("Creating embedding tensor and loading to DRAM")
    torch_embedding = torch.randn(embedding_shape, dtype=embedding_dtype)
    embedding_tensor = ttnn.from_torch(
        torch_embedding, dtype=ttnn_dtype_from_torch_dtype(embedding_dtype), layout=ttnn.ROW_MAJOR_LAYOUT
    )
    embedding_tensor = ttnn.to_device(embedding_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Embedding tensor loaded to DRAM with shape {embedding_shape}")

    logger.info("Creating and Running Host Interface with embedding")
    h2d_socket = ttnn.H2DSocket(mesh_device, h2d_core, ttnn.BufferType.L1, token_fifo_size, h2d_mode)
    d2h_socket = ttnn.D2HSocket(mesh_device, d2h_core, embedding_fifo_size)
    host_io = HostInterface(
        h2d_socket,
        d2h_socket,
        token_size_bytes,
        embedding_size_bytes,
        core_to_core_socket_buffer_size=embedding_fifo_size,
        h2d_downstream_core=fwd_core_0,
        d2h_upstream_core=fwd_core_13,
        embedding_tensor=embedding_tensor,
    )

    socket_interface_1 = SocketInterface(
        embedding_size_bytes,
        embedding_fifo_size,
        embedding_size_bytes,
        fwd_core_0,
        fwd_core_1,
        upstream_socket=host_io.get_downstream_socket(),
        downstream_core_coord=fwd_core_2,
        mesh_device=mesh_device,
    )
    socket_interface_2 = SocketInterface(
        embedding_size_bytes,
        embedding_fifo_size,
        embedding_size_bytes,
        fwd_core_2,
        fwd_core_3,
        upstream_socket=socket_interface_1.get_downstream_socket(),
        downstream_core_coord=fwd_core_4,
        mesh_device=mesh_device,
    )
    socket_interface_3 = SocketInterface(
        embedding_size_bytes,
        embedding_fifo_size,
        embedding_size_bytes,
        fwd_core_4,
        fwd_core_5,
        upstream_socket=socket_interface_2.get_downstream_socket(),
        downstream_core_coord=fwd_core_6,
        mesh_device=mesh_device,
    )
    socket_interface_4 = SocketInterface(
        embedding_size_bytes,
        embedding_fifo_size,
        embedding_size_bytes,
        fwd_core_6,
        fwd_core_7,
        upstream_socket=socket_interface_3.get_downstream_socket(),
        downstream_core_coord=fwd_core_8,
        mesh_device=mesh_device,
    )
    socket_interface_5 = SocketInterface(
        embedding_size_bytes,
        embedding_fifo_size,
        embedding_size_bytes,
        fwd_core_8,
        fwd_core_9,
        upstream_socket=socket_interface_4.get_downstream_socket(),
        downstream_core_coord=fwd_core_10,
        mesh_device=mesh_device,
    )
    socket_interface_6 = SocketInterface(
        embedding_size_bytes,
        embedding_fifo_size,
        embedding_size_bytes,
        fwd_core_10,
        fwd_core_11,
        upstream_socket=socket_interface_5.get_downstream_socket(),
        downstream_core_coord=fwd_core_12,
        mesh_device=mesh_device,
    )
    socket_interface_7 = SocketInterface(
        embedding_size_bytes,
        embedding_fifo_size,
        embedding_size_bytes,
        fwd_core_12,
        fwd_core_13,
        upstream_socket=socket_interface_6.get_downstream_socket(),
        downstream_socket=host_io.get_upstream_socket(),
    )

    host_io.run()
    socket_interface_1.run()
    socket_interface_2.run()
    socket_interface_3.run()
    socket_interface_4.run()
    socket_interface_5.run()
    socket_interface_6.run()
    socket_interface_7.run()

    embedding_row_elems = embedding_shape[3]
    logger.info(f"Testing embedding with vocab size {vocab_size} over multi-stage pipeline")

    for token_id in range(vocab_size):
        # Write 64-byte packet with token ID as first uint32, rest zeros
        torch_input = torch.zeros(1, token_size_datums, dtype=token_dtype)
        torch_input[0, 0] = token_id
        input_tensor = ttnn.from_torch(
            torch_input, dtype=ttnn_dtype_from_torch_dtype(token_dtype), layout=ttnn.ROW_MAJOR_LAYOUT
        )

        # Read embedding row back over D2H socket
        torch_output = torch.zeros(1, embedding_row_elems, dtype=embedding_dtype)
        output_tensor = ttnn.from_torch(
            torch_output, dtype=ttnn_dtype_from_torch_dtype(embedding_dtype), layout=ttnn.ROW_MAJOR_LAYOUT
        )

        h2d_socket.write_tensor(input_tensor)
        d2h_socket.read_tensor(output_tensor)

        result_torch = ttnn.to_torch(output_tensor).reshape(-1)
        expected = torch_embedding[0, 0, token_id, :].reshape(-1)
        match = torch.equal(expected, result_torch)
        assert match, (
            f"Token {token_id}: D2H output does not match embedding row!\n"
            f"Expected: {expected[:8]}...\nGot: {result_torch[:8]}..."
        )

    logger.info(f"{vocab_size} token lookups verified successfully over multi-stage pipeline")

    host_io.terminate(False)
    socket_interface_1.terminate(False)
    socket_interface_2.terminate(False)
    socket_interface_3.terminate(False)
    socket_interface_4.terminate(False)
    socket_interface_5.terminate(False)
    socket_interface_6.terminate(False)
    socket_interface_7.terminate(True)

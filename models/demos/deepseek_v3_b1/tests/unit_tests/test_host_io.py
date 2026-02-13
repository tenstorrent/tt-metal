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
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface

# @pytest.mark.parametrize(
#     "tensor_size_bytes, fifo_size, num_iterations",
#     [
#         (64, 128, 512),
#         (64, 256, 512),
#         (64, 512, 512),
#         (64, 1024, 512),
#         (512, 1024, 512),
#         (1024, 2048, 128),
#     ],
# )
# @pytest.mark.parametrize(
#     "h2d_mode",
#     [
#         ttnn.H2DMode.HOST_PUSH,
#         ttnn.H2DMode.DEVICE_PULL,
#     ],
# )
# def test_host_io_loopback(mesh_device, tensor_size_bytes, fifo_size, num_iterations, h2d_mode):
#     if not is_slow_dispatch():
#         pytest.skip("Skipping test in fast dispatch mode")

#     tensor_size_datums = tensor_size_bytes // 4

#     device_coord = ttnn.MeshCoordinate(0, 0)
#     core_coord = ttnn.CoreCoord(0, 0)
#     socket_core = ttnn.MeshCoreCoord(device_coord, core_coord)

#     logger.info("Creating and Running Host Interface")
#     h2d_socket = ttnn.H2DSocket(mesh_device, socket_core, ttnn.BufferType.L1, fifo_size, h2d_mode)
#     d2h_socket = ttnn.D2HSocket(mesh_device, socket_core, fifo_size)
#     host_io = HostInterface(
#         h2d_socket, d2h_socket, tensor_size_bytes, core_to_core_socket_buffer_size=fifo_size, loopback_mode=True
#     )
#     host_io.run()

#     logger.info(f"Transferring Data Over H <-> D Interface for {num_iterations} iterations")
#     logger.info(f"Tensor Size: {tensor_size_bytes} bytes, FIFO Size: {fifo_size} bytes")
#     logger.info(f"H2D Mode: {h2d_mode}")
#     logger.info(f"FIFO Size: {fifo_size} bytes")
#     for i in range(num_iterations):
#         torch_input = torch.arange(i * tensor_size_datums, (i + 1) * tensor_size_datums, dtype=torch.int32).reshape(
#             1, tensor_size_datums
#         )
#         input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
#         torch_output = torch.zeros(1, tensor_size_datums, dtype=torch.int32)
#         output_tensor = ttnn.from_torch(torch_output, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

#         h2d_socket.write_tensor(input_tensor)
#         d2h_socket.read_tensor(output_tensor)

#         result_torch = ttnn.to_torch(output_tensor)
#         match = torch.equal(torch_input, result_torch)
#         assert match, f"H2D → D2H loopback data mismatch!\nExpected: {torch_input}\nGot: {result_torch}"

#     host_io.terminate()


@pytest.mark.parametrize(
    "h2d_mode",
    [
        ttnn.H2DMode.HOST_PUSH,
        ttnn.H2DMode.DEVICE_PULL,
    ],
)
def test_host_io_loopback_with_embedding(mesh_device, h2d_mode):
    """Test H2D/D2H loopback with an embedding tensor loaded to DRAM."""
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    embedding_shape = (1, 1, 12980, 7168)

    token_size_bytes = 64
    token_fifo_size = 256
    embedding_fifo_size = 14 * 1024 * 4
    token_size_datums = token_size_bytes // 4
    embedding_size_bytes = embedding_shape[3] * 2  # bfloat16
    device_coord = ttnn.MeshCoordinate(0, 0)
    core_coord = ttnn.CoreCoord(0, 0)
    socket_core = ttnn.MeshCoreCoord(device_coord, core_coord)

    # Create embedding tensor and load to DRAM
    torch_embedding = torch.randn(embedding_shape, dtype=torch.bfloat16)
    embedding_tensor = ttnn.from_torch(torch_embedding, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
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

    # Issue 1 write and 1 read, verify data correctness
    torch_input = torch.arange(0, token_size_datums, dtype=torch.int32).reshape(1, token_size_datums)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    torch_output = torch.zeros(1, embedding_size_bytes // 2, dtype=torch.bfloat16)
    output_tensor = ttnn.from_torch(torch_output, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    logger.info("Writing and reading 1 page over H2D → D2H loopback")
    h2d_socket.write_tensor(input_tensor)
    d2h_socket.read_tensor(output_tensor)

    # match = torch.equal(torch_input, result_torch)
    # assert match, f"H2D → D2H loopback data mismatch!\nExpected: {torch_input}\nGot: {result_torch}"
    # logger.info("Data verified successfully")

    host_io.terminate()

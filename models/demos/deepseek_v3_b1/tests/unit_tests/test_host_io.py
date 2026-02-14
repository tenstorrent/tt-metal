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
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size, ttnn_dtype_from_torch_dtype

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
        (512, 8),
    ],
)
def test_host_io_loopback_with_embedding(
    mesh_device, h2d_mode, vocab_size, embedding_dim, token_fifo_size, embedding_fifo_factor
):
    """Test H2D/D2H loopback with an embedding tensor loaded to DRAM."""
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

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

    # Send token IDs 0-127, read back embedding rows, verify against PyTorch reference
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

    host_io.terminate()

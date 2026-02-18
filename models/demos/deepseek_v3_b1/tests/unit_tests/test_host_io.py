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

    host_io.terminate()


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

    host_io.terminate()


@pytest.mark.parametrize(
    "num_h2d_cores",
    [8],
)
@pytest.mark.parametrize(
    "tensor_size_bytes, fifo_size",
    [
        (64, 256),
    ],
)
def test_host_io_multicore_with_sharded_tensor(mesh_device, num_h2d_cores, tensor_size_bytes, fifo_size):
    """Test multi-core H2D to D2H communication with actual sharded tensor data."""
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    tensor_size_datums = tensor_size_bytes // 4

    device_coord = ttnn.MeshCoordinate(0, 0)

    # D2H socket on core (0, 2)
    d2h_core = ttnn.CoreCoord(0, 2)
    d2h_socket_core = ttnn.MeshCoreCoord(device_coord, d2h_core)

    # H2D cores in a 2x4 grid: (0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1)
    h2d_cores = []
    for row in range(2):
        for col in range(4):
            h2d_cores.append(ttnn.MeshCoreCoord(device_coord, ttnn.CoreCoord(col, row)))

    logger.info(f"Creating sharded tensor across {num_h2d_cores} H2D cores")
    logger.info(f"D2H core: {d2h_core}, H2D cores: {[c.core_coord for c in h2d_cores]}")

    # Create test data - one row per H2D core
    # Shape: [1, 1, num_h2d_cores, tensor_size_datums]
    torch_input = torch.zeros(1, 1, num_h2d_cores, tensor_size_datums, dtype=torch.int32)
    for core_idx in range(num_h2d_cores):
        # Mark each shard with its core index
        torch_input[0, 0, core_idx, 0] = core_idx  # Core ID marker
        for j in range(1, tensor_size_datums):
            torch_input[0, 0, core_idx, j] = core_idx * 1000 + j

    logger.info(f"Input tensor shape: {torch_input.shape}")

    # Convert to TTNN tensor and shard across the H2D cores
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Create sharded memory config
    shard_shape = [1, tensor_size_datums]  # List of ints, not ttnn.Shape
    shard_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 1))])
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    input_tensor = ttnn.to_device(input_tensor, mesh_device, memory_config=sharded_mem_config)

    logger.info(f"Created sharded tensor with memory config: {input_tensor.memory_config()}")

    # Create host_io with the sharded tensor
    h2d_socket = ttnn.H2DSocket(mesh_device, d2h_socket_core, ttnn.BufferType.L1, fifo_size, ttnn.H2DMode.HOST_PUSH)
    d2h_socket = ttnn.D2HSocket(mesh_device, d2h_socket_core, fifo_size * num_h2d_cores)

    host_io = HostInterface(
        h2d_socket,
        d2h_socket,
        tensor_size_bytes,
        tensor_size_bytes,
        core_to_core_socket_buffer_size=fifo_size,
        loopback_mode=False,
        num_h2d_cores=num_h2d_cores,
        h2d_cores=h2d_cores,
        input_tensor=input_tensor,
    )
    host_io.run()

    logger.info(f"Testing multi-core data transfer - expecting {num_h2d_cores} pages in round-robin order")

    # Read data in round-robin order from D2H
    for i in range(num_h2d_cores):
        torch_output = torch.zeros(1, tensor_size_datums, dtype=torch.int32)
        output_tensor = ttnn.from_torch(torch_output, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

        d2h_socket.read_tensor(output_tensor)

        result_torch = ttnn.to_torch(output_tensor)
        core_idx = result_torch[0, 0].item()  # Should match round-robin order

        logger.info(f"Read {i}: Got data from core {core_idx}")

        # Verify this is the expected core in round-robin order
        expected_core_idx = i % num_h2d_cores
        assert core_idx == expected_core_idx, f"Round-robin mismatch: expected core {expected_core_idx}, got {core_idx}"

        # Verify data content
        expected_data = torch_input[0, 0, core_idx, :]
        match = torch.equal(expected_data, result_torch[0])
        assert match, f"Data mismatch for core {core_idx}!"

    logger.info(f"✅ Multi-core test passed! All {num_h2d_cores} cores sent data correctly in round-robin order")

    host_io.terminate()

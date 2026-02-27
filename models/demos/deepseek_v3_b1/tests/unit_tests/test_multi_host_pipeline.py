# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Multi-Host Pipeline Block Integration Tests.
Combine H <-> D Interface with Mult-Host sockets under PipelineBlock API.

"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.micro_ops.d2d_exchange.op import MeshWrapper, SocketInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size, ttnn_dtype_from_torch_dtype
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlock


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
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
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
def test_multi_host_loopback_pipeline(mesh_device, tensor_size_bytes, fifo_size, num_iterations, h2d_mode):
    """Test multi-stage pipeline with embedding: H2D receives token, looks up embedding, streams through all devices, D2H sends embedding row back."""
    pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(mesh_device)

    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    my_mesh_id = mesh_device.get_system_mesh_id()

    is_pipeline_start = my_mesh_id == 0

    num_procs = int(ttnn.distributed_context_get_size())
    # Number of pipeline stages is equal to the number of processes + 1 for the loopback stage
    assert len(pipeline_config) == num_procs + 1

    pipeline_core_coord = ttnn.CoreCoord(0, 0)

    if is_pipeline_start:
        # Initialize HostIO interface
        h2d_device_coord = pipeline_config[my_mesh_id].entry_node_coord
        d2h_device_coord = pipeline_config[num_procs].exit_node_coord

        logger.debug(
            f"Creating Host IO Interface on First Pipeline stage. H2D Coord: {h2d_device_coord} D2H Coord: {d2h_device_coord}, H2D Drainer: {pipeline_config[my_mesh_id].exit_node_coord}, D2H Source: {pipeline_config[num_procs].entry_node_coord}"
        )
        logger.debug(f"H2D Device Id: {mesh_device.get_device_id(h2d_device_coord)}")
        logger.debug(f"H2D Drainer Device Id: {mesh_device.get_device_id(pipeline_config[my_mesh_id].exit_node_coord)}")

        h2d_socket = ttnn.H2DSocket(
            mesh_device,
            ttnn.MeshCoreCoord(h2d_device_coord, pipeline_core_coord),
            ttnn.BufferType.L1,
            fifo_size,
            h2d_mode,
        )
        d2h_socket = ttnn.D2HSocket(mesh_device, ttnn.MeshCoreCoord(d2h_device_coord, pipeline_core_coord), fifo_size)

        host_io = HostInterface(
            h2d_socket,
            d2h_socket,
            tensor_size_bytes,
            tensor_size_bytes,
            core_to_core_socket_buffer_size=fifo_size,
            h2d_downstream_core=ttnn.MeshCoreCoord(
                pipeline_config[my_mesh_id].exit_node_coord, pipeline_core_coord
            ),  # H2D Socket connects to exit node of current stage
            d2h_upstream_core=ttnn.MeshCoreCoord(
                pipeline_config[num_procs].entry_node_coord, pipeline_core_coord
            ),  # D2H Socket connects to entry node of last stage
        )

        # Initialize socket interfaces
        logger.debug(
            f"Creating Exit Socket Interface on First Pipeline stage. Exit Coord: {pipeline_config[my_mesh_id].exit_node_coord}, Next Entry Coord: {pipeline_config[my_mesh_id + 1].entry_node_coord}"
        )
        exit_socket_interface = SocketInterface(
            tensor_size_bytes,
            fifo_size,
            tensor_size_bytes,
            ttnn.MeshCoreCoord(
                pipeline_config[my_mesh_id].exit_node_coord, pipeline_core_coord
            ),  # Connects exit node of current stage to entry node of next stage
            ttnn.MeshCoreCoord(pipeline_config[my_mesh_id + 1].entry_node_coord, pipeline_core_coord),
            upstream_socket=host_io.get_downstream_socket(),  # Drains data from H2D
            sender_mesh=MeshWrapper(mesh_device),
            receiver_mesh=MeshWrapper(mesh_id=my_mesh_id + 1),
        )
        logger.debug(
            f"Creating Entry Socket Interface on Last Pipeline stage. Entry Coord: {pipeline_config[num_procs].entry_node_coord}, Prev Exit Coord: {pipeline_config[num_procs - 1].exit_node_coord}"
        )
        entry_socket_interface = SocketInterface(
            tensor_size_bytes,
            fifo_size,
            tensor_size_bytes,
            ttnn.MeshCoreCoord(
                pipeline_config[num_procs - 1].exit_node_coord, pipeline_core_coord
            ),  # Connects exit node of last stage to loopback stage
            ttnn.MeshCoreCoord(pipeline_config[num_procs].entry_node_coord, pipeline_core_coord),
            downstream_socket=host_io.get_upstream_socket(),  # Feeds data to D2H
            sender_mesh=MeshWrapper(mesh_id=num_procs - 1),
            receiver_mesh=MeshWrapper(mesh_device),
        )
        host_io.run()
        exit_socket_interface.run()
        entry_socket_interface.run()

        tensor_size_datums = tensor_size_bytes // 4
        for i in range(num_iterations):
            torch_input = torch.arange(
                i * tensor_size_datums, (i + 1) * tensor_size_datums, dtype=torch.float32
            ).reshape(1, tensor_size_datums)

            input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
            torch_output = torch.zeros(1, tensor_size_datums, dtype=torch.float32)
            output_tensor = ttnn.from_torch(torch_output, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)

            h2d_socket.write_tensor(input_tensor)
            d2h_socket.read_tensor(output_tensor)

            result_torch = ttnn.to_torch(output_tensor)
            match = torch.equal(torch_input, result_torch)
            assert match, f"H2D → D2H loopback data mismatch!\nExpected: {torch_input}\nGot: {result_torch}"

        ttnn.distributed_context_barrier()
        host_io.terminate(False)
        exit_socket_interface.terminate(False)
        entry_socket_interface.terminate(True)

    else:
        logger.debug(
            f"Creating Entry Socket Interface on Pipeline stage {my_mesh_id}. Entry Coord: {pipeline_config[my_mesh_id].entry_node_coord}, Prev Exit Coord: {pipeline_config[my_mesh_id - 1].exit_node_coord}"
        )
        entry_socket_interface = SocketInterface(
            tensor_size_bytes,
            fifo_size,
            tensor_size_bytes,
            ttnn.MeshCoreCoord(pipeline_config[my_mesh_id - 1].exit_node_coord, pipeline_core_coord),
            ttnn.MeshCoreCoord(pipeline_config[my_mesh_id].entry_node_coord, pipeline_core_coord),
            downstream_core_coord=ttnn.MeshCoreCoord(pipeline_config[my_mesh_id].exit_node_coord, pipeline_core_coord),
            sender_mesh=MeshWrapper(mesh_id=my_mesh_id - 1),
            receiver_mesh=MeshWrapper(mesh_device),
        )
        logger.debug(
            f"Creating Exit Socket Interface on Pipeline stage {my_mesh_id}. Exit Coord: {pipeline_config[my_mesh_id].exit_node_coord}, Next Entry Coord: {pipeline_config[my_mesh_id + 1].entry_node_coord}"
        )
        exit_socket_interface = SocketInterface(
            tensor_size_bytes,
            fifo_size,
            tensor_size_bytes,
            ttnn.MeshCoreCoord(pipeline_config[my_mesh_id].exit_node_coord, pipeline_core_coord),
            ttnn.MeshCoreCoord(pipeline_config[my_mesh_id + 1].entry_node_coord, pipeline_core_coord),
            upstream_socket=entry_socket_interface.get_downstream_socket(),
            sender_mesh=MeshWrapper(mesh_device),
            receiver_mesh=MeshWrapper(mesh_id=my_mesh_id + 1 if my_mesh_id < num_procs - 1 else 0),
        )
        entry_socket_interface.run()
        exit_socket_interface.run()

        ttnn.distributed_context_barrier()

        entry_socket_interface.terminate(False)
        exit_socket_interface.terminate(True)


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
        (256, 14336),
        (512, 7168),
        (1024, 3584),
        (2048, 1792),
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
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
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
def test_multi_host_loopback_pipeline_with_embedding(
    mesh_device, h2d_mode, vocab_size, embedding_dim, token_fifo_size, embedding_fifo_factor
):
    """Test multi-host pipeline with embedding: H2D receives token, looks up embedding row, streams it through pipeline stages across hosts, D2H sends embedding row back."""
    pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(mesh_device)

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

    my_mesh_id = mesh_device.get_system_mesh_id()
    is_pipeline_start = my_mesh_id == 0

    num_procs = int(ttnn.distributed_context_get_size())
    assert len(pipeline_config) == num_procs + 1

    pipeline_core_coord = ttnn.CoreCoord(0, 0)

    if is_pipeline_start:
        h2d_device_coord = pipeline_config[my_mesh_id].entry_node_coord
        d2h_device_coord = pipeline_config[num_procs].exit_node_coord

        logger.debug(
            f"Creating Host IO Interface with embedding on First Pipeline stage. "
            f"H2D Coord: {h2d_device_coord} D2H Coord: {d2h_device_coord}, "
            f"H2D Drainer: {pipeline_config[my_mesh_id].exit_node_coord}, "
            f"D2H Source: {pipeline_config[num_procs].entry_node_coord}"
        )

        # Create embedding tensor and load to DRAM
        logger.info("Creating embedding tensor and loading to DRAM")
        torch_embedding = torch.randn(embedding_shape, dtype=embedding_dtype)
        embedding_tensor = ttnn.from_torch(
            torch_embedding, dtype=ttnn_dtype_from_torch_dtype(embedding_dtype), layout=ttnn.ROW_MAJOR_LAYOUT
        )
        embedding_tensor = ttnn.to_device(embedding_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        logger.info(f"Embedding tensor loaded to DRAM with shape {embedding_shape}")

        h2d_socket = ttnn.H2DSocket(
            mesh_device,
            ttnn.MeshCoreCoord(h2d_device_coord, pipeline_core_coord),
            ttnn.BufferType.L1,
            token_fifo_size,
            h2d_mode,
        )
        d2h_socket = ttnn.D2HSocket(
            mesh_device, ttnn.MeshCoreCoord(d2h_device_coord, pipeline_core_coord), embedding_fifo_size
        )

        host_io = HostInterface(
            h2d_socket,
            d2h_socket,
            token_size_bytes,
            embedding_size_bytes,
            core_to_core_socket_buffer_size=embedding_fifo_size,
            h2d_downstream_core=ttnn.MeshCoreCoord(pipeline_config[my_mesh_id].exit_node_coord, pipeline_core_coord),
            d2h_upstream_core=ttnn.MeshCoreCoord(pipeline_config[num_procs].entry_node_coord, pipeline_core_coord),
            embedding_tensor=embedding_tensor,
        )

        logger.debug(
            f"Creating Exit Socket Interface on First Pipeline stage. "
            f"Exit Coord: {pipeline_config[my_mesh_id].exit_node_coord}, "
            f"Next Entry Coord: {pipeline_config[my_mesh_id + 1].entry_node_coord}"
        )
        exit_socket_interface = SocketInterface(
            embedding_size_bytes,
            embedding_fifo_size,
            embedding_size_bytes,
            ttnn.MeshCoreCoord(pipeline_config[my_mesh_id].exit_node_coord, pipeline_core_coord),
            ttnn.MeshCoreCoord(pipeline_config[my_mesh_id + 1].entry_node_coord, pipeline_core_coord),
            upstream_socket=host_io.get_downstream_socket(),
            sender_mesh=MeshWrapper(mesh_device),
            receiver_mesh=MeshWrapper(mesh_id=my_mesh_id + 1),
        )
        logger.debug(
            f"Creating Entry Socket Interface on Last Pipeline stage. "
            f"Entry Coord: {pipeline_config[num_procs].entry_node_coord}, "
            f"Prev Exit Coord: {pipeline_config[num_procs - 1].exit_node_coord}"
        )
        entry_socket_interface = SocketInterface(
            embedding_size_bytes,
            embedding_fifo_size,
            embedding_size_bytes,
            ttnn.MeshCoreCoord(pipeline_config[num_procs - 1].exit_node_coord, pipeline_core_coord),
            ttnn.MeshCoreCoord(pipeline_config[num_procs].entry_node_coord, pipeline_core_coord),
            downstream_socket=host_io.get_upstream_socket(),
            sender_mesh=MeshWrapper(mesh_id=num_procs - 1),
            receiver_mesh=MeshWrapper(mesh_device),
        )
        host_io.run()
        exit_socket_interface.run()
        entry_socket_interface.run()

        embedding_row_elems = embedding_shape[3]
        logger.info(f"Testing embedding with vocab size {vocab_size} over multi-host pipeline")

        for token_id in range(vocab_size):
            torch_input = torch.zeros(1, token_size_datums, dtype=token_dtype)
            torch_input[0, 0] = token_id
            input_tensor = ttnn.from_torch(
                torch_input, dtype=ttnn_dtype_from_torch_dtype(token_dtype), layout=ttnn.ROW_MAJOR_LAYOUT
            )

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

        logger.info(f"{vocab_size} token lookups verified successfully over multi-host pipeline")

        ttnn.distributed_context_barrier()
        host_io.terminate(False)
        exit_socket_interface.terminate(False)
        entry_socket_interface.terminate(True)

    else:
        logger.debug(
            f"Creating Entry Socket Interface on Pipeline stage {my_mesh_id}. "
            f"Entry Coord: {pipeline_config[my_mesh_id].entry_node_coord}, "
            f"Prev Exit Coord: {pipeline_config[my_mesh_id - 1].exit_node_coord}"
        )
        entry_socket_interface = SocketInterface(
            embedding_size_bytes,
            embedding_fifo_size,
            embedding_size_bytes,
            ttnn.MeshCoreCoord(pipeline_config[my_mesh_id - 1].exit_node_coord, pipeline_core_coord),
            ttnn.MeshCoreCoord(pipeline_config[my_mesh_id].entry_node_coord, pipeline_core_coord),
            downstream_core_coord=ttnn.MeshCoreCoord(pipeline_config[my_mesh_id].exit_node_coord, pipeline_core_coord),
            sender_mesh=MeshWrapper(mesh_id=my_mesh_id - 1),
            receiver_mesh=MeshWrapper(mesh_device),
        )
        logger.debug(
            f"Creating Exit Socket Interface on Pipeline stage {my_mesh_id}. "
            f"Exit Coord: {pipeline_config[my_mesh_id].exit_node_coord}, "
            f"Next Entry Coord: {pipeline_config[my_mesh_id + 1].entry_node_coord}"
        )
        exit_socket_interface = SocketInterface(
            embedding_size_bytes,
            embedding_fifo_size,
            embedding_size_bytes,
            ttnn.MeshCoreCoord(pipeline_config[my_mesh_id].exit_node_coord, pipeline_core_coord),
            ttnn.MeshCoreCoord(pipeline_config[my_mesh_id + 1].entry_node_coord, pipeline_core_coord),
            upstream_socket=entry_socket_interface.get_downstream_socket(),
            sender_mesh=MeshWrapper(mesh_device),
            receiver_mesh=MeshWrapper(mesh_id=my_mesh_id + 1 if my_mesh_id < num_procs - 1 else 0),
        )
        entry_socket_interface.run()
        exit_socket_interface.run()

        ttnn.distributed_context_barrier()

        entry_socket_interface.terminate(False)
        exit_socket_interface.terminate(True)


@pytest.mark.parametrize(
    "vocab_size, embedding_dim",
    [
        (256, 14336),
        (512, 7168),
        (1024, 3584),
        (2048, 1792),
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
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
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
def test_pipeline_block(mesh_device, vocab_size, embedding_dim, token_fifo_size, embedding_fifo_factor):
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    pipeline_core_coord = ttnn.CoreCoord(0, 0)

    embedding_dtype = torch.bfloat16
    embedding_shape = (1, 1, vocab_size, embedding_dim)
    embedding_size_bytes = embedding_dim * dtype_size(embedding_dtype)
    embedding_fifo_size = embedding_size_bytes * embedding_fifo_factor

    if mesh_device.get_system_mesh_id() == 0:
        torch_embedding = torch.randn(embedding_shape, dtype=embedding_dtype)
        embedding_tensor = ttnn.from_torch(
            torch_embedding, dtype=ttnn_dtype_from_torch_dtype(embedding_dtype), layout=ttnn.ROW_MAJOR_LAYOUT
        )
        embedding_tensor = ttnn.to_device(embedding_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        pipeline_block = PipelineBlock(
            mesh_device,
            pipeline_core_coord,
            embedding_fifo_size,  # upstream d2d socket fifo size
            embedding_fifo_size,  # downstream d2d socket fifo size
            embedding_size_bytes,  # upstream d2d socket page size
            embedding_size_bytes,  # downstream d2d socket page size
            h2d_socket_fifo_size=token_fifo_size,  # h2d socket fifo size
            d2h_socket_fifo_size=embedding_fifo_size,  # d2h socket fifo size
            d2h_socket_page_size=embedding_size_bytes,  # d2h socket page size
            embedding_tensor=embedding_tensor,
        )
    else:
        pipeline_block = PipelineBlock(
            mesh_device,
            pipeline_core_coord,
            embedding_fifo_size,  # upstream d2d socket fifo size
            embedding_fifo_size,  # downstream d2d socket fifo size
            embedding_size_bytes,  # upstream d2d socket page size
            embedding_size_bytes,  # downstream d2d socket page size
        )

    pipeline_block.run()

    if pipeline_block.is_first_pipeline_stage():
        token_dtype = torch.uint32
        token_size_bytes = 64
        token_size_datums = token_size_bytes // dtype_size(token_dtype)

        for token_id in range(vocab_size):
            torch_input = torch.zeros(1, token_size_datums, dtype=token_dtype)
            torch_input[0, 0] = token_id
            input_tensor = ttnn.from_torch(
                torch_input, dtype=ttnn_dtype_from_torch_dtype(token_dtype), layout=ttnn.ROW_MAJOR_LAYOUT
            )
            torch_output = torch.zeros(1, embedding_shape[3], dtype=embedding_dtype)
            output_tensor = ttnn.from_torch(
                torch_output, dtype=ttnn_dtype_from_torch_dtype(embedding_dtype), layout=ttnn.ROW_MAJOR_LAYOUT
            )
            pipeline_block.write_token(input_tensor)
            pipeline_block.read_output(output_tensor)

            result_torch = ttnn.to_torch(output_tensor).reshape(-1)
            expected = torch_embedding[0, 0, token_id, :].reshape(-1)
            match = torch.equal(expected, result_torch)
            assert match, (
                f"Token {token_id}: D2H output does not match embedding row!\n"
                f"Expected: {expected[:8]}...\nGot: {result_torch[:8]}..."
            )
        logger.info(f"{vocab_size} token lookups verified successfully over multi-host pipeline")

    pipeline_block.terminate()

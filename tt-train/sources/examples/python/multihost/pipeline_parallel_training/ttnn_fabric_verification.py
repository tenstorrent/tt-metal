# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time
import torch
import ttnn
from loguru import logger


def run_ttnn_fabric_verification(num_rows: int = 4, num_cols: int = 8) -> None:
    """
    Optional TTNN fabric verification routine.

    This helper contains the verification logic for send/recv between ranks. It is not invoked by default.
    The number of rows and cols passed in must match the MGD you are using.

    Currently, the function just tests forward connections for a ring topology, but this could be extende
    in the future by modifying connections_to_test.
    """

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    mesh_shape = ttnn.MeshShape(num_rows, num_cols)

    device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

    if not ttnn.distributed_context_is_initialized():
        raise RuntimeError(f"Distributed context failed to initialize!")

    world_size = int(ttnn.distributed_context_get_size())
    rank = int(ttnn.distributed_context_get_rank())

    logger.info(f"Rank {rank}/{world_size} initialized successfully")
    torch.manual_seed(42)

    socket_connections = [
        ttnn.SocketConnection(
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)),
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)),
        )
    ]
    socket_mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 4096)

    test_shape = (1, 1, 32, 32)
    torch_tensor = torch.randn(test_shape, dtype=torch.bfloat16)

    connections_to_test = [(i, (i + 1) % world_size) for i in range(world_size)]

    for sender, receiver in connections_to_test:
        ttnn.distributed_context_barrier()
        logger.info(f"Rank {rank} passed barrier, starting transfer test")

        if rank == sender:
            print(f"Rank {rank} is sending data")
            tt_tensor = ttnn.from_torch(
                torch_tensor,
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )

            logger.info(f"Rank {rank} trying to send to rank {receiver}")
            socket_config = ttnn.SocketConfig(
                socket_connections,
                socket_mem_config,
                sender_rank=sender,
                receiver_rank=receiver,
            )
            send_socket = ttnn.MeshSocket(device, socket_config)
            logger.info(f"Rank {rank} sending tensor of shape {test_shape}")

            t_start = time.time()
            ttnn.experimental.send_async(tt_tensor, send_socket)
            ttnn.synchronize_device(device)
            t_end = time.time()

            logger.info(
                f"Rank {rank} send completed in {(t_end - t_start)*1000:.2f} ms"
            )
            del send_socket
            del tt_tensor

        if rank == receiver:
            print(f"Rank {rank} is receiving data")

            padded_shape = [1, 1, 32, 32]
            recv_tensor = ttnn.allocate_tensor_on_device(
                ttnn.TensorSpec(padded_shape, ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT),
                device,
            )

            logger.info(f"Rank {rank} trying to receive from rank {sender}")
            socket_config = ttnn.SocketConfig(
                socket_connections,
                socket_mem_config,
                sender_rank=sender,
                receiver_rank=receiver,
            )
            recv_socket = ttnn.MeshSocket(device, socket_config)

            logger.info(f"Rank {rank} waiting to receive tensor")

            t_start = time.time()
            ttnn.experimental.recv_async(recv_tensor, recv_socket)
            ttnn.synchronize_device(device)
            t_end = time.time()

            logger.info(
                f"Rank {rank} receive completed in {(t_end - t_start)*1000:.2f} ms"
            )

            received_torch = ttnn.to_torch(
                recv_tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0)
            )[0]

            if torch.allclose(received_torch, torch_tensor, rtol=1e-2, atol=1e-2):
                logger.success(f"Data verification PASSED - tensor matches!")
            else:
                max_diff = (received_torch - torch_tensor).abs().max().item()
                logger.error(f"Data verification FAILED - max diff: {max_diff}")
                logger.error(f"expected {torch_tensor}, received {received_torch}")
                raise ValueError("Data mismatch between sent and received tensors")

            del recv_socket
            del recv_tensor

    ttnn.distributed_context_barrier()
    logger.info(f"Rank {rank} test completed successfully")

    ttnn.close_device(device)

    return


if __name__ == "__main__":
    run_ttnn_fabric_verification(num_rows=1, num_cols=8)

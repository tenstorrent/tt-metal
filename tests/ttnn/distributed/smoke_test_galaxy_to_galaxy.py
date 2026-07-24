#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Smoke test for Galaxy-to-Galaxy communication on Exabox cluster.

This test verifies basic connectivity between two Galaxy systems by:
1. Establishing a socket connection between the two galaxies
2. Sending a small tensor from Galaxy 0 to Galaxy 1
3. Verifying the received data matches

Run with:
    tt-run --rank-binding tests/tt_metal/distributed/config/exabox_2_galaxy_rank_binding.yaml \
        --mpi-args "--tag-output" \
        python tests/ttnn/distributed/smoke_test_galaxy_to_galaxy.py
"""

import os
import socket
import time
import torch
import ttnn
from loguru import logger


def get_node_info():
    """Get hostname and process info for logging."""
    hostname = socket.gethostname()
    pid = os.getpid()
    return hostname, pid


def run_smoke_test():
    hostname, pid = get_node_info()
    logger.info(f"Starting smoke test on {hostname} (PID: {pid})")

    torch.manual_seed(42)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    mesh_shape = ttnn.MeshShape(8, 4)
    logger.info(f"{hostname}: Opening mesh device with shape {mesh_shape}")

    device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

    if not ttnn.distributed_context_is_initialized():
        raise RuntimeError(f"{hostname}: Distributed context failed to initialize!")

    world_size = int(ttnn.distributed_context_get_size())
    rank = int(ttnn.distributed_context_get_rank())

    logger.info(f"{hostname}: Rank {rank}/{world_size} initialized successfully")

    if world_size != 2:
        raise ValueError(f"Expected 2 processes, got {world_size}")

    socket_connections = [
        ttnn.SocketConnection(
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)),
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)),
        )
    ]
    socket_mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 4096)
    socket_config = ttnn.SocketConfig(socket_connections, socket_mem_config, sender_rank=0, receiver_rank=1)

    test_shape = (1, 1, 32, 32)
    torch_tensor = torch.randn(test_shape, dtype=torch.bfloat16)

    ttnn.distributed_context_barrier()
    logger.info(f"{hostname}: Rank {rank} passed barrier, starting transfer test")

    if rank == 0:
        tt_tensor = ttnn.from_torch(
            torch_tensor,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

        send_socket = ttnn.MeshSocket(device, socket_config)
        logger.info(f"{hostname}: Rank 0 sending tensor of shape {test_shape}")

        t_start = time.time()
        ttnn.experimental.send_async(tt_tensor, send_socket)
        ttnn.synchronize_device(device)
        t_end = time.time()

        logger.info(f"{hostname}: Rank 0 send completed in {(t_end - t_start)*1000:.2f} ms")
        del send_socket
        del tt_tensor

    else:
        recv_socket = ttnn.MeshSocket(device, socket_config)
        padded_shape = [1, 1, 32, 32]
        recv_tensor = ttnn.allocate_tensor_on_device(
            ttnn.TensorSpec(padded_shape, ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT), device
        )

        logger.info(f"{hostname}: Rank 1 waiting to receive tensor")

        t_start = time.time()
        ttnn.experimental.recv_async(recv_tensor, recv_socket)
        ttnn.synchronize_device(device)
        t_end = time.time()

        logger.info(f"{hostname}: Rank 1 receive completed in {(t_end - t_start)*1000:.2f} ms")

        received_torch = ttnn.to_torch(recv_tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[0]

        if torch.allclose(received_torch, torch_tensor, rtol=1e-2, atol=1e-2):
            logger.success(f"{hostname}: Data verification PASSED - tensor matches!")
        else:
            max_diff = (received_torch - torch_tensor).abs().max().item()
            logger.error(f"{hostname}: Data verification FAILED - max diff: {max_diff}")
            raise ValueError("Data mismatch between sent and received tensors")

        del recv_socket
        del recv_tensor

    ttnn.distributed_context_barrier()
    logger.info(f"{hostname}: Rank {rank} test completed successfully")

    ttnn.close_device(device)

    print(f"\n{'='*60}")
    print(f"SMOKE TEST PASSED on {hostname} (Rank {rank})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_smoke_test()

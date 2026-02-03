#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test: Transfer a SHARDED tensor from one N300 to another N300.

Physical topology (from system-health.txt):
  - Rank 0 N300: Chip 0 (PCIe 1) + Chip 4 (internal trace)
  - Rank 1 N300: Chip 2 (PCIe 3) + Chip 6 (internal trace)
  - Direct QSFP: Chip 0 ↔ Chip 2
  - Fabric routing: Chip 4 → Chip 0 → Chip 2 → Chip 6

With socket connections on BOTH chips, shards transfer directly:

    Source N300 (rank 0)                    Target N300 (rank 1)
    +---------------------+                 +---------------------+
    |  Chip 0   |  Chip 4 |                 |  Chip 2   |  Chip 6 |
    | [shard 0] |[shard 1]|   send_async    | [shard 0] |[shard 1]|
    |     |     |    |    |  ----------->   |     |     |    |    |
    |     +-----+----+----+-----------------+-----+-----+----+    |
    |           |         |  (fabric routes)|           |         |
    +-----------+---------+                 +-----------+---------+

Run with:
tt-run --rank-binding tests/ttnn/distributed/disaggregated_pd_rank_binding.yaml \
    python tests/ttnn/distributed/simple_sharded_n300_transfer.py
"""

import torch
import ttnn
from loguru import logger

SOCKET_FIFO_SIZE = 1024 * 1024
TENSOR_SHAPE = (2, 1, 64, 128)  # Small tensor for testing


def create_predefined_tensor():
    """Create a predefined tensor with known values for testing."""
    # Create a tensor with a simple pattern that's easy to verify
    torch_tensor = torch.arange(
        TENSOR_SHAPE[0] * TENSOR_SHAPE[1] * TENSOR_SHAPE[2] * TENSOR_SHAPE[3],
        dtype=torch.bfloat16,
    ).reshape(TENSOR_SHAPE)
    return torch_tensor


def setup_socket(device, sender_rank=0, receiver_rank=1):
    """
    Setup socket for cross-N300 communication with connections on BOTH chips.

    - Connection 1: Chip 0 (0,0) ↔ Chip 2 (0,0) - direct QSFP
    - Connection 2: Chip 4 (0,1) ↔ Chip 6 (0,1) - routed via fabric

    This enables direct shard-to-shard transfer without gather/scatter!
    """
    socket_connections = [
        # Chip 0 ↔ Chip 2 (direct QSFP link)
        ttnn.SocketConnection(
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 6)),
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 6)),
        ),
        # Chip 4 ↔ Chip 6 (fabric routes through Chip 0 → Chip 2)
        # ttnn.SocketConnection(
        #     ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 1), ttnn.CoreCoord(0, 7)),
        #     ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 1), ttnn.CoreCoord(0, 7)),
        # ),
    ]
    socket_mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, SOCKET_FIFO_SIZE)
    return ttnn.SocketConfig(
        socket_connections, socket_mem_config, sender_rank=sender_rank, receiver_rank=receiver_rank
    )


def run_sender(device, socket, rank):
    """
    Sender side: Create a sharded tensor and send directly.

    With socket connections on both chips, each chip sends its shard directly:
    - Chip 0 sends shard 0 → Chip 2
    - Chip 4 sends shard 1 → Chip 6 (routed via fabric)
    """
    logger.info(f"Rank {rank}: === Sending SHARDED tensor ===")

    torch_tensor = create_predefined_tensor()
    logger.info(f"Rank {rank}: Original tensor shape: {torch_tensor.shape}")

    # Create sharded tensor across both chips
    tt_sharded = ttnn.from_torch(
        torch_tensor,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),
    )
    logger.info(f"Rank {rank}: Sharded tensor created")

    # Extract shards by mesh coordinate
    # device_tensors[0] = mesh coordinate (0,0), device_tensors[1] = mesh coordinate (0,1)
    device_tensors = ttnn.get_device_tensors(tt_sharded)
    mesh_coords = [(0, 0), (0, 1)]  # Corresponding to device_tensors indices

    for i, (mesh_coord, dt) in enumerate(zip(mesh_coords, device_tensors)):
        chip_data = ttnn.to_torch(dt)
        logger.info(f"Rank {rank}: Mesh {mesh_coord} shard: {chip_data.flatten()[:4].tolist()}")

    # Send sharded tensor directly - no gather needed!
    logger.info(f"Rank {rank}: Sending sharded tensor (each chip sends its shard)...")
    ttnn.experimental.send_async(tt_sharded, socket)
    ttnn.synchronize_device(device)
    logger.info(f"Rank {rank}: Send complete!")

    del tt_sharded


def run_receiver(device, socket, rank):
    """
    Receiver side: Receive sharded tensor directly.

    With socket connections on both chips, each chip receives its shard directly:
    - Chip 2 receives shard 0 from Chip 0
    - Chip 6 receives shard 1 from Chip 4 (routed via fabric)
    """
    logger.info(f"Rank {rank}: === Receiving tensor ===")

    # Each chip receives one shard - shape is (1, 1, H, W) not (2, 1, H, W)
    shard_shape = [
        TENSOR_SHAPE[0] // 2,  # Each chip gets half
        TENSOR_SHAPE[1],
        ((TENSOR_SHAPE[2] + 31) // 32) * 32,
        ((TENSOR_SHAPE[3] + 31) // 32) * 32,
    ]
    logger.info(f"Rank {rank}: Receive buffer shape per chip: {shard_shape}")

    recv_tensor = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(shard_shape, ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT), device
    )

    logger.info(f"Rank {rank}: Receiving sharded tensor (each chip receives its shard)...")
    ttnn.experimental.recv_async(recv_tensor, socket)
    ttnn.synchronize_device(device)
    logger.info(f"Rank {rank}: Received! Shape: {recv_tensor.shape}")

    # Extract received shards by mesh coordinate
    # device_tensors[0] = mesh coordinate (0,0), device_tensors[1] = mesh coordinate (0,1)
    device_tensors = ttnn.get_device_tensors(recv_tensor)
    mesh_coords = [(0, 0), (0, 1)]  # Corresponding to device_tensors indices
    logger.info(f"Rank {rank}: Number of device tensors: {len(device_tensors)}")

    receiver_shards = {}
    for i, (mesh_coord, dt) in enumerate(zip(mesh_coords, device_tensors)):
        chip_data = ttnn.to_torch(dt)
        receiver_shards[mesh_coord] = chip_data
        logger.info(f"Rank {rank}: Mesh {mesh_coord} received shard: {chip_data.flatten()[:4].tolist()}")

    # Generate expected shards from the same predefined tensor
    logger.info(f"Rank {rank}: === VERIFICATION ===")
    torch_tensor = create_predefined_tensor()
    expected_tt_sharded = ttnn.from_torch(
        torch_tensor,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),
    )
    expected_device_tensors = ttnn.get_device_tensors(expected_tt_sharded)
    expected_shards = {}
    for i, (mesh_coord, dt) in enumerate(zip(mesh_coords, expected_device_tensors)):
        expected_shards[mesh_coord] = ttnn.to_torch(dt)

    # Compare received shards with expected shards by mesh coordinate
    all_match = True
    for mesh_coord in mesh_coords:
        if mesh_coord in expected_shards and mesh_coord in receiver_shards:
            expected_shard = expected_shards[mesh_coord]
            receiver_shard = receiver_shards[mesh_coord]

            # Compare shards
            if torch.allclose(expected_shard, receiver_shard, rtol=1e-5, atol=1e-5):
                logger.info(f"Rank {rank}: SUCCESS - Mesh {mesh_coord}: Expected and received shards match!")
            else:
                logger.error(f"Rank {rank}: FAILED - Mesh {mesh_coord}: Expected and received shards do NOT match!")
                logger.error(f"Rank {rank}:   Expected shard (first 4): {expected_shard.flatten()[:4].tolist()}")
                logger.error(f"Rank {rank}:   Received shard (first 4): {receiver_shard.flatten()[:4].tolist()}")
                all_match = False
        else:
            logger.warning(f"Rank {rank}: WARNING - Mesh {mesh_coord} not found in expected or received shards")
            all_match = False

    if all_match:
        logger.info(f"Rank {rank}: === OVERALL SUCCESS - All mesh coordinate comparisons passed! ===")
    else:
        logger.error(f"Rank {rank}: === OVERALL FAILURE - Some mesh coordinate comparisons failed! ===")

    del recv_tensor
    del expected_tt_sharded


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    mesh_shape = ttnn.MeshShape(1, 2)  # N300 has 2 chips
    visible_device_ids = ttnn.get_device_ids()

    if len(visible_device_ids) != 2:
        raise ValueError(f"Expected 2 devices (N300), got {len(visible_device_ids)}")

    device = ttnn.open_mesh_device(mesh_shape=mesh_shape, physical_device_ids=visible_device_ids)

    if not ttnn.distributed_context_is_initialized():
        raise ValueError("Distributed context not initialized. Use tt-run with rank binding.")

    world_size = int(ttnn.distributed_context_get_size())
    if world_size != 2:
        raise ValueError(f"Requires 2 processes (2 N300s), got {world_size}")

    rank = int(ttnn.distributed_context_get_rank())
    logger.info(f"Rank {rank}: Started with {len(visible_device_ids)} devices")

    socket_config = setup_socket(device)
    socket = ttnn.MeshSocket(device, socket_config)
    ttnn.distributed_context_barrier()

    if rank == 0:
        run_sender(device, socket, rank)
    else:
        run_receiver(device, socket, rank)

    del socket
    ttnn.distributed_context_barrier()
    ttnn.close_device(device)
    logger.info(f"Rank {rank}: Finished")


if __name__ == "__main__":
    main()

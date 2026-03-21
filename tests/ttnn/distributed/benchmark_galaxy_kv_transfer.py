#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Galaxy-to-Galaxy KV Cache Transfer Benchmark for DeepSeek V3

Benchmarks KV cache transfers between different mesh shape configurations:
- 1x8 → 1x8 (row to row)
- 1x8 → 4x2 (row to block)
- 4x2 → 1x8 (block to row)

Run with:
tt-run --rank-binding tests/tt_metal/distributed/config/exabox_2_galaxy_rank_binding.yaml \
    --mpi-args "--hostfile <hostfile> --mca btl_tcp_if_exclude docker0,lo --tag-output" \
    python tests/ttnn/distributed/benchmark_galaxy_kv_transfer.py
"""

import argparse
import os
import socket
import statistics
import time
import json
import torch
import ttnn
from loguru import logger
from typing import Tuple, List, Dict, Any

# DeepSeek V3 KV cache parameters
NUM_LAYERS = 61
KVPE_DIM = 576
BLOCK_SIZE = 32
SOCKET_FIFO_SIZE = 1024 * 1024

SEQ_LENGTHS = [1024, 4096, 8192, 32768, 131072]
NUM_WARMUP = 2
NUM_TRANSFERS = 5


def get_kv_cache_shape(seq_len: int) -> Tuple[int, int, int, int]:
    """Get KV cache shape for a given sequence length."""
    num_blocks = seq_len // BLOCK_SIZE
    return (num_blocks, 1, BLOCK_SIZE, KVPE_DIM)


def calculate_size_bytes(shape: Tuple[int, ...]) -> int:
    """Calculate tensor size in bytes (bfloat8_b = 1 byte per element)."""
    size = 1
    for dim in shape:
        size *= dim
    return size


def setup_socket(device, sender_mesh_shape, receiver_mesh_shape, sender_rank=0, receiver_rank=1):
    """
    Create socket configuration for cross-galaxy transfer.
    1 connection: (0,0)→(0,0)
    """
    socket_connections = [
        ttnn.SocketConnection(
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)),
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)),
        ),
    ]

    logger.info(f"Created {len(socket_connections)} socket connection: (0,0)→(0,0)")

    socket_mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, SOCKET_FIFO_SIZE)
    return ttnn.SocketConfig(
        socket_connections, socket_mem_config, sender_rank=sender_rank, receiver_rank=receiver_rank
    )


def run_e2e_benchmark_sender(
    device, socket, seq_len: int, num_transfers: int, num_devices: int = 8
) -> Tuple[List[float], float, float, float]:
    """
    Run end-to-end benchmark: send all layers, measure total time.
    Tensor is sharded across the mesh along dimension 0.
    Returns: (times_ms, size_mb_total, size_mb_per_layer, shard_size_mb)
    """
    shape = get_kv_cache_shape(seq_len)
    size_bytes_per_layer = calculate_size_bytes(shape)
    size_mb_per_layer = size_bytes_per_layer / (1024 * 1024)
    size_mb_total = size_mb_per_layer * NUM_LAYERS
    shard_size_mb = size_mb_per_layer / num_devices

    torch_tensor = torch.randn(shape, dtype=torch.bfloat16)
    tt_tensor = ttnn.from_torch(
        torch_tensor,
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),
    )

    times = []
    for _ in range(num_transfers):
        ttnn.distributed_context_barrier()

        t_start = time.time_ns()
        for _ in range(NUM_LAYERS):
            ttnn.experimental.send_async(tt_tensor, socket)
        ttnn.synchronize_device(device)
        ttnn.distributed_context_barrier()
        t_end = time.time_ns()

        times.append((t_end - t_start) / 1_000_000)

    del tt_tensor
    return times, size_mb_total, size_mb_per_layer, shard_size_mb


def run_e2e_benchmark_receiver(device, socket, seq_len: int, num_transfers: int, num_devices: int = 8):
    """Receiver side: receive all layers. Each device receives its shard."""
    shape = get_kv_cache_shape(seq_len)
    shard_dim0 = shape[0] // num_devices
    shard_shape = [shard_dim0, shape[1], ((shape[2] + 31) // 32) * 32, ((shape[3] + 31) // 32) * 32]

    for _ in range(num_transfers):
        ttnn.distributed_context_barrier()

        for _ in range(NUM_LAYERS):
            recv_tensor = ttnn.allocate_tensor_on_device(
                ttnn.TensorSpec(shard_shape, ttnn.DataType.BFLOAT8_B, ttnn.TILE_LAYOUT), device
            )
            ttnn.experimental.recv_async(recv_tensor, socket)
        ttnn.synchronize_device(device)
        ttnn.distributed_context_barrier()

        del recv_tensor


def warmup(device, socket, rank: int, mesh_shape):
    """Warmup transfers to prime the socket."""
    num_devices = mesh_shape[0] * mesh_shape[1]
    shape = (32 * num_devices, 1, 32, KVPE_DIM)
    shard_shape = [32, 1, 32, ((KVPE_DIM + 31) // 32) * 32]

    for _ in range(NUM_WARMUP):
        if rank == 0:
            torch_tensor = torch.randn(shape, dtype=torch.bfloat16)
            tt_tensor = ttnn.from_torch(
                torch_tensor,
                device=device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),
            )
            ttnn.experimental.send_async(tt_tensor, socket)
            ttnn.synchronize_device(device)
            del tt_tensor
        else:
            recv_tensor = ttnn.allocate_tensor_on_device(
                ttnn.TensorSpec(shard_shape, ttnn.DataType.BFLOAT8_B, ttnn.TILE_LAYOUT), device
            )
            ttnn.experimental.recv_async(recv_tensor, socket)
            ttnn.synchronize_device(device)
            del recv_tensor
        ttnn.distributed_context_barrier()


def compute_stats(times: List[float]) -> Dict[str, float]:
    """Compute statistics for timing data."""
    return {
        "min": min(times),
        "max": max(times),
        "avg": statistics.mean(times),
        "median": statistics.median(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
    }


def run_benchmark(sender_mesh_shape: Tuple[int, int], receiver_mesh_shape: Tuple[int, int]):
    """Run the full benchmark suite."""
    hostname = socket.gethostname()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    rank = int(os.environ.get("TT_MESH_ID", "0"))
    mesh_shape = sender_mesh_shape if rank == 0 else receiver_mesh_shape

    logger.info(f"[{hostname}] Rank {rank}: Opening mesh with shape {mesh_shape}")
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape))

    if not ttnn.distributed_context_is_initialized():
        raise RuntimeError("Distributed context not initialized")

    world_size = int(ttnn.distributed_context_get_size())
    if world_size != 2:
        raise ValueError(f"Requires 2 processes, got {world_size}")

    rank = int(ttnn.distributed_context_get_rank())
    logger.info(f"[{hostname}] Rank {rank}/{world_size}: Device opened, mesh_shape={mesh_shape}")

    socket_config = setup_socket(device, sender_mesh_shape, receiver_mesh_shape)
    mesh_socket = ttnn.MeshSocket(device, socket_config)
    ttnn.distributed_context_barrier()

    logger.info(f"[{hostname}] Rank {rank}: Running warmup...")
    warmup(device, mesh_socket, rank, mesh_shape)
    logger.info(f"[{hostname}] Rank {rank}: Warmup complete")

    results = {
        "config": {
            "sender_mesh": list(sender_mesh_shape),
            "receiver_mesh": list(receiver_mesh_shape),
            "num_layers": NUM_LAYERS,
            "kvpe_dim": KVPE_DIM,
            "block_size": BLOCK_SIZE,
            "num_transfers": NUM_TRANSFERS,
        },
        "benchmarks": [],
    }

    num_devices = sender_mesh_shape[0] * sender_mesh_shape[1]

    if rank == 0:
        print("\n" + "=" * 160)
        print(f"DEEPSEEK V3 KV CACHE TRANSFER BENCHMARK (Sharded)")
        print(
            f"Sender Mesh: {sender_mesh_shape[0]}x{sender_mesh_shape[1]} → Receiver Mesh: {receiver_mesh_shape[0]}x{receiver_mesh_shape[1]}"
        )
        print(f"Layers: {NUM_LAYERS}, KVPE_DIM: {KVPE_DIM}, Transfers: {NUM_TRANSFERS}, Devices: {num_devices}")
        print(f"Socket connections: 1 ((0,0)→(0,0))")
        print("=" * 160)
        print(
            f"{'Seq Len':<10} {'Total MB':<10} {'Layer MB':<10} {'Shard MB':<10} {'Avg(ms)':<12} {'Median(ms)':<12} "
            f"{'Min(ms)':<12} {'Max(ms)':<12} {'Throughput':<14} {'Per Layer':<12}"
        )
        print("-" * 160)

    for seq_len in SEQ_LENGTHS:
        if rank == 0:
            times, size_mb_total, size_mb_per_layer, shard_size_mb = run_e2e_benchmark_sender(
                device, mesh_socket, seq_len, NUM_TRANSFERS, num_devices
            )

            stats = compute_stats(times)
            avg_per_layer_ms = stats["avg"] / NUM_LAYERS
            median_per_layer_ms = stats["median"] / NUM_LAYERS

            throughput_gbs = (size_mb_total / 1024) / (stats["avg"] / 1000) if stats["avg"] > 0 else 0
            throughput_per_layer_gbs = (
                (size_mb_per_layer / 1024) / (avg_per_layer_ms / 1000) if avg_per_layer_ms > 0 else 0
            )

            result = {
                "seq_len": seq_len,
                "size_mb_total": size_mb_total,
                "size_mb_per_layer": size_mb_per_layer,
                "shard_size_mb": shard_size_mb,
                "times_ms": times,
                "stats": stats,
                "avg_per_layer_ms": avg_per_layer_ms,
                "median_per_layer_ms": median_per_layer_ms,
                "throughput_total_gbs": throughput_gbs,
                "throughput_per_layer_gbs": throughput_per_layer_gbs,
            }
            results["benchmarks"].append(result)

            print(
                f"{seq_len:<10} {size_mb_total:<10.1f} {size_mb_per_layer:<10.2f} {shard_size_mb:<10.2f} "
                f"{stats['avg']:<12.1f} {stats['median']:<12.1f} "
                f"{stats['min']:<12.1f} {stats['max']:<12.1f} "
                f"{throughput_gbs:<14.2f} {throughput_per_layer_gbs:<12.2f}"
            )
        else:
            run_e2e_benchmark_receiver(device, mesh_socket, seq_len, NUM_TRANSFERS, num_devices)

    if rank == 0:
        print("=" * 160)
        print(f"\nPer-layer timing (Avg ms / Median ms) | Shard size per device:")
        for r in results["benchmarks"]:
            print(
                f"  seq_len={r['seq_len']:<6}: {r['avg_per_layer_ms']:.3f} ms / {r['median_per_layer_ms']:.3f} ms | shard={r['shard_size_mb']:.2f} MB"
            )

        results_file = f"/data/dmadic/kv_benchmark_{sender_mesh_shape[0]}x{sender_mesh_shape[1]}_to_{receiver_mesh_shape[0]}x{receiver_mesh_shape[1]}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    del mesh_socket
    ttnn.distributed_context_barrier()
    ttnn.close_device(device)
    logger.info(f"[{hostname}] Rank {rank}: Finished")


def main():
    parser = argparse.ArgumentParser(description="Galaxy KV Cache Transfer Benchmark")
    parser.add_argument("--sender-mesh", type=str, default="1x8", help="Sender mesh shape (e.g., 1x8, 4x2)")
    parser.add_argument("--receiver-mesh", type=str, default="1x8", help="Receiver mesh shape (e.g., 1x8, 4x2)")
    args = parser.parse_args()

    def parse_mesh_shape(s: str) -> Tuple[int, int]:
        parts = s.lower().split("x")
        return (int(parts[0]), int(parts[1]))

    sender_mesh = parse_mesh_shape(args.sender_mesh)
    receiver_mesh = parse_mesh_shape(args.receiver_mesh)

    logger.info(f"Starting benchmark: {sender_mesh} → {receiver_mesh}")
    run_benchmark(sender_mesh, receiver_mesh)


if __name__ == "__main__":
    main()

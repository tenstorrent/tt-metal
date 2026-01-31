#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Run with:
tt-run --rank-binding tests/ttnn/distributed/disaggregated_pd_rank_binding.yaml \
    python tests/ttnn/distributed/benchmark_deepseek_v3_kv_transfer.py
"""

import time
import json
import torch
import ttnn
from loguru import logger

NUM_LAYERS = 61
KVPE_DIM = 576
BLOCK_SIZE = 32
SOCKET_FIFO_SIZE = 1024 * 1024

SEQ_LENGTHS = [1024, 4096, 8192, 32768, 163840]
SEQ_LENGTHS_SINGLE_TENSOR = [1024, 4096, 8192, 32768]
NUM_TRANSFERS = 3
MAX_SINGLE_TENSOR_MB = 2048

RESULTS_FILE = "/localdev/dmadic/tt-metal/tests/ttnn/distributed/deepseek_v3_kv_benchmark_results.json"


def get_kv_cache_shape(seq_len):
    num_blocks = seq_len // BLOCK_SIZE
    return (num_blocks, 1, BLOCK_SIZE, KVPE_DIM)


def calculate_size_bytes(shape):
    return shape[0] * shape[1] * shape[2] * shape[3]


def setup_socket(device, mesh_shape, sender_rank=0, receiver_rank=1):
    socket_connections = [
        ttnn.SocketConnection(
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 6)),
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 6)),
        ),
        ttnn.SocketConnection(
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 7)),
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 7)),
        ),
    ]
    socket_mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, SOCKET_FIFO_SIZE)
    return ttnn.SocketConfig(
        socket_connections, socket_mem_config, sender_rank=sender_rank, receiver_rank=receiver_rank
    )


def run_sender_single_tensor(device, socket, seq_len):
    shape = get_kv_cache_shape(seq_len)
    total_shape = (shape[0] * NUM_LAYERS, shape[1], shape[2], shape[3])
    size_bytes = calculate_size_bytes(total_shape)
    size_mb = size_bytes / (1024 * 1024)

    torch_tensor = torch.randn(total_shape, dtype=torch.bfloat16)
    tt_tensor = ttnn.from_torch(
        torch_tensor,
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    times = []
    for i in range(NUM_TRANSFERS):
        t_start = time.time_ns()
        ttnn.experimental.send_async(tt_tensor, socket)
        ttnn.synchronize_device(device)
        t_end = time.time_ns()
        times.append((t_end - t_start) / 1_000_000)

    del tt_tensor
    return times, size_mb


def run_sender_layer_by_layer(device, socket, seq_len):
    shape = get_kv_cache_shape(seq_len)
    size_bytes_per_layer = calculate_size_bytes(shape)
    size_mb_per_layer = size_bytes_per_layer / (1024 * 1024)
    size_mb_total = size_mb_per_layer * NUM_LAYERS

    torch_tensor = torch.randn(shape, dtype=torch.bfloat16)
    tt_tensor = ttnn.from_torch(
        torch_tensor,
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    total_times = []
    per_layer_times = []
    for i in range(NUM_TRANSFERS):
        layer_times = []
        t_total_start = time.time_ns()
        for _ in range(NUM_LAYERS):
            t_layer_start = time.time_ns()
            ttnn.experimental.send_async(tt_tensor, socket)
            ttnn.synchronize_device(device)
            t_layer_end = time.time_ns()
            layer_times.append((t_layer_end - t_layer_start) / 1_000_000)
        t_total_end = time.time_ns()
        total_times.append((t_total_end - t_total_start) / 1_000_000)
        per_layer_times.append(layer_times)

    del tt_tensor
    return total_times, per_layer_times, size_mb_total, size_mb_per_layer


def run_receiver_single_tensor(device, socket, seq_len):
    shape = get_kv_cache_shape(seq_len)
    total_shape = (shape[0] * NUM_LAYERS, shape[1], shape[2], shape[3])
    padded_shape = [
        total_shape[0],
        total_shape[1],
        ((total_shape[2] + 31) // 32) * 32,
        ((total_shape[3] + 31) // 32) * 32,
    ]

    for _ in range(NUM_TRANSFERS):
        recv_tensor = ttnn.allocate_tensor_on_device(
            ttnn.TensorSpec(padded_shape, ttnn.DataType.BFLOAT8_B, ttnn.TILE_LAYOUT), device
        )
        ttnn.experimental.recv_async(recv_tensor, socket)
        ttnn.synchronize_device(device)
        del recv_tensor


def run_receiver_layer_by_layer(device, socket, seq_len):
    shape = get_kv_cache_shape(seq_len)
    padded_shape = [shape[0], shape[1], ((shape[2] + 31) // 32) * 32, ((shape[3] + 31) // 32) * 32]

    for _ in range(NUM_TRANSFERS):
        for _ in range(NUM_LAYERS):
            recv_tensor = ttnn.allocate_tensor_on_device(
                ttnn.TensorSpec(padded_shape, ttnn.DataType.BFLOAT8_B, ttnn.TILE_LAYOUT), device
            )
            ttnn.experimental.recv_async(recv_tensor, socket)
            ttnn.synchronize_device(device)
            del recv_tensor


def warmup_sender(device, socket):
    shape = (32, 1, 32, KVPE_DIM)
    torch_tensor = torch.randn(shape, dtype=torch.bfloat16)
    tt_tensor = ttnn.from_torch(
        torch_tensor,
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    ttnn.experimental.send_async(tt_tensor, socket)
    ttnn.synchronize_device(device)
    del tt_tensor


def warmup_receiver(device, socket):
    shape = [32, 1, 32, ((KVPE_DIM + 31) // 32) * 32]
    recv_tensor = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(shape, ttnn.DataType.BFLOAT8_B, ttnn.TILE_LAYOUT), device
    )
    ttnn.experimental.recv_async(recv_tensor, socket)
    ttnn.synchronize_device(device)
    del recv_tensor


def send_termination(device, socket):
    term = torch.tensor([[[[0]]]], dtype=torch.int64)
    tt_term = ttnn.from_torch(
        term, device=device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
    )
    ttnn.experimental.send_async(tt_term, socket)
    ttnn.synchronize_device(device)


def recv_termination(device, socket):
    recv = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec([1, 1, 32, 32], ttnn.DataType.UINT32, ttnn.TILE_LAYOUT), device
    )
    ttnn.experimental.recv_async(recv, socket)
    ttnn.synchronize_device(device)
    del recv


def run_benchmark():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    mesh_shape = ttnn.MeshShape(1, 2)
    visible_device_ids = ttnn.get_device_ids()

    if len(visible_device_ids) != 2:
        raise ValueError(f"Expected 2 devices, got {len(visible_device_ids)}")

    device = ttnn.open_mesh_device(mesh_shape=mesh_shape, physical_device_ids=visible_device_ids)

    if not ttnn.distributed_context_is_initialized():
        raise ValueError("Distributed context not initialized")

    world_size = int(ttnn.distributed_context_get_size())
    if world_size != 2:
        raise ValueError(f"Requires 2 processes, got {world_size}")

    rank = int(ttnn.distributed_context_get_rank())
    logger.info(f"Rank {rank}: Started")

    socket_config = setup_socket(device, mesh_shape)
    socket = ttnn.MeshSocket(device, socket_config)
    ttnn.distributed_context_barrier()

    logger.info(f"Rank {rank}: Running warmup transfer...")
    if rank == 0:
        warmup_sender(device, socket)
    else:
        warmup_receiver(device, socket)
    ttnn.distributed_context_barrier()
    logger.info(f"Rank {rank}: Warmup complete")

    results = {"single_tensor": [], "layer_by_layer": []}

    if rank == 0:
        logger.info("=== SINGLE TENSOR MODE (skipping 163840 - too large for host memory) ===")
        for seq_len in SEQ_LENGTHS_SINGLE_TENSOR:
            times, size_mb = run_sender_single_tensor(device, socket, seq_len)
            avg_ms = sum(times) / len(times)
            throughput = (size_mb / 1024) / (avg_ms / 1000) if avg_ms > 0 else 0
            results["single_tensor"].append(
                {
                    "seq_len": seq_len,
                    "size_mb": size_mb,
                    "avg_ms": avg_ms,
                    "throughput_gbs": throughput,
                    "times_ms": times,
                }
            )
            logger.info(f"  seq_len={seq_len:>6}: {size_mb:>8.1f} MB, {avg_ms:>8.1f} ms, {throughput:>6.2f} GB/s")
        send_termination(device, socket)

        ttnn.distributed_context_barrier()

        logger.info("=== LAYER BY LAYER MODE ===")
        for seq_len in SEQ_LENGTHS:
            total_times, per_layer_times, size_mb_total, size_mb_per_layer = run_sender_layer_by_layer(
                device, socket, seq_len
            )
            avg_total_ms = sum(total_times) / len(total_times)
            all_layer_times = [t for run in per_layer_times for t in run]
            avg_per_layer_ms = sum(all_layer_times) / len(all_layer_times)
            throughput_total = (size_mb_total / 1024) / (avg_total_ms / 1000) if avg_total_ms > 0 else 0
            throughput_per_layer = (size_mb_per_layer / 1024) / (avg_per_layer_ms / 1000) if avg_per_layer_ms > 0 else 0
            results["layer_by_layer"].append(
                {
                    "seq_len": seq_len,
                    "size_mb_total": size_mb_total,
                    "size_mb_per_layer": size_mb_per_layer,
                    "avg_total_ms": avg_total_ms,
                    "avg_per_layer_ms": avg_per_layer_ms,
                    "throughput_total_gbs": throughput_total,
                    "throughput_per_layer_gbs": throughput_per_layer,
                    "total_times_ms": total_times,
                    "per_layer_times_ms": per_layer_times,
                }
            )
            logger.info(
                f"  seq_len={seq_len:>6}: total={size_mb_total:>7.1f} MB in {avg_total_ms:>7.1f} ms ({throughput_total:>5.2f} GB/s), "
                f"per_layer={size_mb_per_layer:>5.1f} MB in {avg_per_layer_ms:>5.2f} ms ({throughput_per_layer:>5.2f} GB/s)"
            )
        send_termination(device, socket)

        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {RESULTS_FILE}")

        print_summary(results)

    else:
        for seq_len in SEQ_LENGTHS_SINGLE_TENSOR:
            run_receiver_single_tensor(device, socket, seq_len)
        recv_termination(device, socket)

        ttnn.distributed_context_barrier()

        for seq_len in SEQ_LENGTHS:
            run_receiver_layer_by_layer(device, socket, seq_len)
        recv_termination(device, socket)

    del socket
    ttnn.distributed_context_barrier()
    ttnn.close_device(device)
    logger.info(f"Rank {rank}: Finished")


def print_summary(results):
    print("\n" + "=" * 130)
    print("DEEPSEEK V3 KV CACHE TRANSFER BENCHMARK (61 layers, kvpe_dim=576, bfloat8_b)")
    print("=" * 130)
    print(
        f"{'Seq Len':<10} {'Total MB':<10} {'Single(ms)':<12} {'LBL Total(ms)':<14} {'LBL/Layer(ms)':<14} "
        f"{'Single GB/s':<12} {'LBL Tot GB/s':<13} {'LBL Lyr GB/s':<12}"
    )
    print("-" * 130)

    single_dict = {r["seq_len"]: r for r in results["single_tensor"]}
    for lbl in results["layer_by_layer"]:
        seq_len = lbl["seq_len"]
        if seq_len in single_dict:
            s = single_dict[seq_len]
            print(
                f"{seq_len:<10} {lbl['size_mb_total']:<10.1f} {s['avg_ms']:<12.1f} {lbl['avg_total_ms']:<14.1f} {lbl['avg_per_layer_ms']:<14.2f} "
                f"{s['throughput_gbs']:<12.2f} {lbl['throughput_total_gbs']:<13.2f} {lbl['throughput_per_layer_gbs']:<12.2f}"
            )
        else:
            print(
                f"{seq_len:<10} {lbl['size_mb_total']:<10.1f} {'N/A':<12} {lbl['avg_total_ms']:<14.1f} {lbl['avg_per_layer_ms']:<14.2f} "
                f"{'N/A':<12} {lbl['throughput_total_gbs']:<13.2f} {lbl['throughput_per_layer_gbs']:<12.2f}"
            )

    print("=" * 130)


if __name__ == "__main__":
    run_benchmark()

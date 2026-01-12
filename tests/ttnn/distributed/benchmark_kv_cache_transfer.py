#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark KV Cache Transfer Latency for Disaggregated Prefill-Decode.

This script measures the latency for transferring KV cache tensors of different sizes
between two N300 devices using TT-Fabric sockets.

Run with:
    cd $TT_METAL_HOME
    tt-run --rank-binding tests/ttnn/distributed/disaggregated_pd_rank_binding.yaml \
           /localdev/dmadic/tt-metal/python_env/bin/python tests/ttnn/distributed/benchmark_kv_cache_transfer.py
"""

import time
import json
import torch
import ttnn
from loguru import logger
from pathlib import Path


# KV cache dimensions to benchmark
# Format: (batch_size, num_kv_heads, seq_len, head_dim)
# Batch=32, varying sequence lengths from 128 to 32K (powers of 2)
# Simulates batched inference with different prompt lengths
BENCHMARK_CONFIGS = [
    {"name": "b32_seq_128", "shape": (32, 8, 128, 128), "num_transfers": 10},
    {"name": "b32_seq_256", "shape": (32, 8, 256, 128), "num_transfers": 10},
    {"name": "b32_seq_512", "shape": (32, 8, 512, 128), "num_transfers": 10},
    {"name": "b32_seq_1024", "shape": (32, 8, 1024, 128), "num_transfers": 8},
    {"name": "b32_seq_2048", "shape": (32, 8, 2048, 128), "num_transfers": 6},
    {"name": "b32_seq_4096", "shape": (32, 8, 4096, 128), "num_transfers": 5},
    {"name": "b32_seq_8192", "shape": (32, 8, 8192, 128), "num_transfers": 4},
    {"name": "b32_seq_16384", "shape": (32, 8, 16384, 128), "num_transfers": 3},
    {"name": "b32_seq_32768", "shape": (32, 8, 32768, 128), "num_transfers": 2},
]

# Results file path
RESULTS_FILE = "/localdev/dmadic/tt-metal/tests/ttnn/distributed/kv_cache_benchmark_results.json"
PLOT_FILE = "/localdev/dmadic/tt-metal/tests/ttnn/distributed/kv_cache_transfer_latency.png"


def setup_socket(device, mesh_shape, sender_rank=0, receiver_rank=1):
    """Setup socket for data transfer between ranks."""
    sender_coord = ttnn.CoreCoord(0, 0)
    recv_coord = ttnn.CoreCoord(0, 0)

    socket_connections = []
    for coord in ttnn.MeshCoordinateRange(mesh_shape):
        socket_connections.append(
            ttnn.SocketConnection(ttnn.MeshCoreCoord(coord, sender_coord), ttnn.MeshCoreCoord(coord, recv_coord))
        )

    socket_mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 16384)

    socket_config = ttnn.SocketConfig(
        socket_connections,
        socket_mem_config,
        sender_rank=sender_rank,
        receiver_rank=receiver_rank,
    )

    return socket_config


def calculate_tensor_size_bytes(shape, dtype=ttnn.bfloat16):
    """Calculate tensor size in bytes."""
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    bytes_per_element = 2 if dtype == ttnn.bfloat16 else 4
    return num_elements * bytes_per_element


def run_sender(device, socket, configs):
    """
    Sender (Rank 0): Create and send tensors of various sizes.
    Send timing info along with each transfer.
    """
    logger.info("=== SENDER (Rank 0) - Starting benchmark ===")

    results = []

    for config in configs:
        name = config["name"]
        shape = config["shape"]
        num_transfers = config["num_transfers"]

        tensor_size_bytes = calculate_tensor_size_bytes(shape)
        tensor_size_mb = tensor_size_bytes / (1024 * 1024)

        logger.info(f"\n--- Benchmarking: {name} ---")
        logger.info(f"Shape: {shape}, Size: {tensor_size_mb:.2f} MB, Transfers: {num_transfers}")

        # Create tensor on device
        torch_tensor = torch.randn(shape, dtype=torch.bfloat16)
        tt_tensor = ttnn.from_torch(
            torch_tensor,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        transfer_times = []

        for i in range(num_transfers):
            # Send timing metadata first (wall-clock nanoseconds)
            t_send_start_ns = time.time_ns()

            # Pack timestamp into tensor (split 64-bit into two 32-bit)
            t_start_high = int(t_send_start_ns >> 32)
            t_start_low = int(t_send_start_ns & 0xFFFFFFFF)

            metadata = torch.tensor(
                [[[[t_start_high, t_start_low, shape[0], shape[1], shape[2], shape[3], 0, 0]]]], dtype=torch.int64
            )
            metadata_tt = ttnn.from_torch(
                metadata,
                device=device,
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
            )

            # Send metadata then data tensor
            ttnn.experimental.send_async(metadata_tt, socket)
            ttnn.experimental.send_async(tt_tensor, socket)

            # Synchronize to ensure transfer completes
            ttnn.synchronize_device(device)

            t_send_end_ns = time.time_ns()
            sender_time_ms = (t_send_end_ns - t_send_start_ns) / 1_000_000
            transfer_times.append(sender_time_ms)

            logger.info(f"  Transfer {i+1}/{num_transfers}: {sender_time_ms:.2f}ms (sender-side)")

        avg_time = sum(transfer_times) / len(transfer_times)
        min_time = min(transfer_times)
        max_time = max(transfer_times)

        results.append(
            {
                "name": name,
                "shape": shape,
                "size_bytes": tensor_size_bytes,
                "size_mb": tensor_size_mb,
                "num_transfers": num_transfers,
                "sender_times_ms": transfer_times,
                "sender_avg_ms": avg_time,
                "sender_min_ms": min_time,
                "sender_max_ms": max_time,
            }
        )

        logger.info(f"  Sender avg: {avg_time:.2f}ms, min: {min_time:.2f}ms, max: {max_time:.2f}ms")

        # Deallocate tensor to free memory
        del tt_tensor
        del metadata_tt

    # Send termination signal (shape with zeros)
    term_metadata = torch.tensor([[[[0, 0, 0, 0, 0, 0, 0, 0]]]], dtype=torch.int64)
    term_metadata_tt = ttnn.from_torch(
        term_metadata,
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn.experimental.send_async(term_metadata_tt, socket)
    ttnn.synchronize_device(device)

    logger.info("\n=== SENDER - Benchmark complete ===")
    return results


def run_receiver(device, socket, configs):
    """
    Receiver (Rank 1): Receive tensors and measure end-to-end latency.
    """
    logger.info("=== RECEIVER (Rank 1) - Starting benchmark ===")

    results = []

    for config in configs:
        name = config["name"]
        shape = config["shape"]
        num_transfers = config["num_transfers"]

        tensor_size_bytes = calculate_tensor_size_bytes(shape)
        tensor_size_mb = tensor_size_bytes / (1024 * 1024)

        logger.info(f"\n--- Receiving: {name} ---")
        logger.info(f"Expected shape: {shape}, Size: {tensor_size_mb:.2f} MB, Transfers: {num_transfers}")

        e2e_times = []
        receiver_times = []

        for i in range(num_transfers):
            t_recv_start = time.perf_counter()

            # Allocate receive buffers
            metadata_recv = ttnn.allocate_tensor_on_device(
                ttnn.TensorSpec([1, 1, 32, 32], ttnn.DataType.UINT32, ttnn.TILE_LAYOUT), device
            )

            # Receive metadata first
            ttnn.experimental.recv_async(metadata_recv, socket)
            ttnn.synchronize_device(device)

            # Parse metadata
            metadata_full = ttnn.to_torch(
                ttnn.from_device(metadata_recv), mesh_composer=ttnn.ConcatMeshToTensor(device, dim=-1)
            )
            metadata = (
                metadata_full[:, :, :, : metadata_full.shape[3] // 2] if metadata_full.shape[3] > 8 else metadata_full
            )

            t_start_high = int(metadata[0, 0, 0, 0].item())
            t_start_low = int(metadata[0, 0, 0, 1].item())

            # Reconstruct sender timestamp
            t_send_start_ns = ((t_start_high & 0xFFFFFFFF) << 32) | (t_start_low & 0xFFFFFFFF)

            # Allocate tensor receive buffer with the expected shape (tile-padded)
            # Tile layout pads to multiples of 32
            padded_shape = [shape[0], shape[1], ((shape[2] + 31) // 32) * 32, ((shape[3] + 31) // 32) * 32]
            data_recv = ttnn.allocate_tensor_on_device(
                ttnn.TensorSpec(padded_shape, ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT), device
            )

            # Receive data tensor
            ttnn.experimental.recv_async(data_recv, socket)
            ttnn.synchronize_device(device)

            t_recv_end_ns = time.time_ns()
            t_recv_end = time.perf_counter()

            # Calculate times
            e2e_time_ms = (t_recv_end_ns - t_send_start_ns) / 1_000_000
            receiver_time_ms = (t_recv_end - t_recv_start) * 1000

            e2e_times.append(e2e_time_ms)
            receiver_times.append(receiver_time_ms)

            logger.info(f"  Transfer {i+1}/{num_transfers}: E2E={e2e_time_ms:.2f}ms, Recv={receiver_time_ms:.2f}ms")

            # Cleanup
            del metadata_recv
            del data_recv

        avg_e2e = sum(e2e_times) / len(e2e_times)
        avg_recv = sum(receiver_times) / len(receiver_times)

        results.append(
            {
                "name": name,
                "shape": shape,
                "size_bytes": tensor_size_bytes,
                "size_mb": tensor_size_mb,
                "num_transfers": num_transfers,
                "e2e_times_ms": e2e_times,
                "e2e_avg_ms": avg_e2e,
                "e2e_min_ms": min(e2e_times),
                "e2e_max_ms": max(e2e_times),
                "receiver_times_ms": receiver_times,
                "receiver_avg_ms": avg_recv,
            }
        )

        throughput_mbps = tensor_size_mb / (avg_e2e / 1000) if avg_e2e > 0 else 0
        logger.info(f"  E2E avg: {avg_e2e:.2f}ms, Throughput: {throughput_mbps:.1f} MB/s")

    # Receive termination signal
    term_recv = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec([1, 1, 32, 32], ttnn.DataType.UINT32, ttnn.TILE_LAYOUT), device
    )
    ttnn.experimental.recv_async(term_recv, socket)
    ttnn.synchronize_device(device)
    del term_recv

    logger.info("\n=== RECEIVER - Benchmark complete ===")
    return results


def generate_plot(results):
    """Generate plot from benchmark results."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Set up the figure with multiple subplots
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("KV Cache Transfer Latency Benchmark\n(TT-Fabric N300 → N300)", fontsize=14, fontweight="bold")

    # Extract data
    names = [r["name"] for r in results]
    sizes_mb = [r["size_mb"] for r in results]
    e2e_avg = [r["e2e_avg_ms"] for r in results]
    e2e_min = [r["e2e_min_ms"] for r in results]
    e2e_max = [r["e2e_max_ms"] for r in results]
    throughputs = [r["size_mb"] / (r["e2e_avg_ms"] / 1000) if r["e2e_avg_ms"] > 0 else 0 for r in results]

    # Color coding by category
    colors = []
    for name in names:
        if "batch" in name:
            colors.append("#e74c3c")  # Red for batch variations
        elif "heads" in name:
            colors.append("#9b59b6")  # Purple for head variations
        elif "llama" in name:
            colors.append("#27ae60")  # Green for Llama configs
        else:
            colors.append("#3498db")  # Blue for seq variations

    # Plot 1: Latency vs Size (scatter)
    ax1 = fig.add_subplot(2, 2, 1)
    scatter = ax1.scatter(sizes_mb, e2e_avg, c=colors, s=100, alpha=0.7, edgecolors="black", linewidth=0.5)
    ax1.set_xlabel("Tensor Size (MB)", fontsize=11)
    ax1.set_ylabel("E2E Latency (ms)", fontsize=11)
    ax1.set_title("Transfer Latency vs Tensor Size", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(sizes_mb, e2e_avg, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(sizes_mb), max(sizes_mb), 100)
    ax1.plot(x_line, p(x_line), "k--", alpha=0.5, label=f"Linear fit")
    ax1.legend()

    # Plot 2: Throughput vs Size
    ax2 = fig.add_subplot(2, 2, 2)
    bars = ax2.bar(range(len(names)), throughputs, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Configuration", fontsize=11)
    ax2.set_ylabel("Throughput (MB/s)", fontsize=11)
    ax2.set_title("Transfer Throughput by Configuration", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    # Add throughput values on bars
    for bar, tp in zip(bars, throughputs):
        height = bar.get_height()
        ax2.annotate(
            f"{tp:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # Plot 3: Latency with error bars
    ax3 = fig.add_subplot(2, 2, 3)
    x_pos = range(len(names))
    yerr_lower = [avg - mn for avg, mn in zip(e2e_avg, e2e_min)]
    yerr_upper = [mx - avg for avg, mx in zip(e2e_avg, e2e_max)]
    ax3.errorbar(
        x_pos,
        e2e_avg,
        yerr=[yerr_lower, yerr_upper],
        fmt="o",
        capsize=4,
        capthick=1.5,
        markersize=8,
        color="#2c3e50",
        ecolor="#7f8c8d",
        elinewidth=1.5,
    )
    ax3.set_xlabel("Configuration", fontsize=11)
    ax3.set_ylabel("E2E Latency (ms)", fontsize=11)
    ax3.set_title("Latency Distribution (min/avg/max)", fontsize=12, fontweight="bold")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Size vs Latency log-log scale
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.loglog(sizes_mb, e2e_avg, "o-", color="#2980b9", markersize=8, linewidth=2, alpha=0.7)
    ax4.set_xlabel("Tensor Size (MB) - Log Scale", fontsize=11)
    ax4.set_ylabel("E2E Latency (ms) - Log Scale", fontsize=11)
    ax4.set_title("Latency vs Size (Log-Log Scale)", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3, which="both")

    # Add annotations for each point
    for i, (x, y, name) in enumerate(zip(sizes_mb, e2e_avg, names)):
        ax4.annotate(name.split("_")[0], (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.7)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#3498db", edgecolor="black", label="Seq length variations"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="Batch size variations"),
        Patch(facecolor="#9b59b6", edgecolor="black", label="Head count variations"),
        Patch(facecolor="#27ae60", edgecolor="black", label="Llama 8B configs"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    logger.info(f"Plot saved to {PLOT_FILE}")

    # Also create a summary table
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    print(f"{'Config':<25} {'Shape':<25} {'Size(MB)':<12} {'E2E(ms)':<12} {'Throughput(MB/s)':<15}")
    print("-" * 100)
    for r in results:
        shape_str = f"({r['shape'][0]},{r['shape'][1]},{r['shape'][2]},{r['shape'][3]})"
        print(
            f"{r['name']:<25} {shape_str:<25} {r['size_mb']:<12.2f} {r['e2e_avg_ms']:<12.2f} {r['size_mb']/(r['e2e_avg_ms']/1000):<15.1f}"
        )
    print("=" * 100)


def run_benchmark():
    """Main entry point for the KV cache transfer benchmark."""

    # Initialize TT-Fabric for inter-device communication
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    # Each process gets one N300 (1x2 mesh, 2 chips per board)
    mesh_shape = ttnn.MeshShape(1, 2)

    # Get visible device IDs
    visible_device_ids = ttnn.get_device_ids()
    if len(visible_device_ids) != 2:
        raise ValueError(
            f"Expected exactly 2 devices for N300 (1x2 mesh), got {len(visible_device_ids)}. "
            f"Visible device IDs: {visible_device_ids}."
        )

    physical_device_ids = visible_device_ids
    device = ttnn.open_mesh_device(mesh_shape=mesh_shape, physical_device_ids=physical_device_ids)

    # Verify distributed context
    if not ttnn.distributed_context_is_initialized():
        raise ValueError("Distributed context not initialized. Run with tt-run.")

    world_size = int(ttnn.distributed_context_get_size())
    if world_size != 2:
        raise ValueError(f"This benchmark requires exactly 2 processes, got {world_size}")

    rank = int(ttnn.distributed_context_get_rank())
    logger.info(f"Process {rank} started on N300 device")

    # Setup socket
    socket_config = setup_socket(device, mesh_shape)

    # Create socket before benchmark
    logger.info(f"Rank {rank}: Creating socket...")
    socket = ttnn.MeshSocket(device, socket_config)
    logger.info(f"Rank {rank}: Socket created")

    # Barrier to ensure both sockets are established
    ttnn.distributed_context_barrier()
    logger.info(f"Rank {rank}: Socket handshake complete")

    # Run benchmark
    if rank == 0:
        results = run_sender(device, socket, BENCHMARK_CONFIGS)
        # Save sender-side results
        with open(RESULTS_FILE + ".sender", "w") as f:
            json.dump(results, f, indent=2)
    else:
        results = run_receiver(device, socket, BENCHMARK_CONFIGS)
        # Save receiver-side results (these have E2E timing)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {RESULTS_FILE}")

        # Generate plot on receiver side (has E2E data)
        generate_plot(results)

    # Cleanup
    del socket
    ttnn.distributed_context_barrier()
    ttnn.close_device(device)

    logger.info(f"Process {rank} finished")


if __name__ == "__main__":
    run_benchmark()

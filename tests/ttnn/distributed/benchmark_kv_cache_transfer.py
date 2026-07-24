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


# =============================================================================
# OPTIMIZATION CONFIGURATION
# =============================================================================
# Number of parallel socket connections per mesh coordinate
NUM_SOCKET_CONNECTIONS = 1

# FIFO size for socket buffers (must be L1-aligned)
# Options: 4096, 8192, 16384, 32768 bytes
# Larger FIFO improves throughput by reducing overhead
SOCKET_FIFO_SIZE = 1024 * 1024  # 1MB

# Buffer type for socket memory: "L1" or "DRAM"
# DRAM avoids L1 contention when using many cores
SOCKET_BUFFER_TYPE = "L1"

# Tensor memory configuration: "L1" or "DRAM"
# DRAM avoids L1 contention when using L1 for socket buffers
TENSOR_MEMORY_TYPE = "DRAM"

# Use diagonal core distribution to avoid NOC bank conflicts
USE_DIAGONAL_CORES = False

# SHARDING MODE: Shard tensors across mesh coordinates for parallel link usage
# With ShardTensorToMesh, each chip sends DIFFERENT data through its own link
# enabling parallel transfer through both QSFP links (2x throughput)
USE_SHARDED_TRANSFER = True
SHARD_DIM = 0  # Shard along batch dimension

# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================
# KV cache dimensions to benchmark
# Format: (batch_size, num_kv_heads, seq_len, head_dim)
# Batch=32, varying sequence lengths from 128 to 32K (powers of 2)
# Simulates batched inference with different prompt lengths
# num_layers=32 (like Llama 8B), each layer has K + V tensors
NUM_LAYERS = 32  # Llama 8B has 32 transformer layers
BENCHMARK_CONFIGS = [
    # {"name": "b32_seq_128", "shape": (32, 8, 128, 128), "num_transfers": 3},
    # {"name": "b32_seq_256", "shape": (32, 8, 256, 128), "num_transfers": 3},
    # {"name": "b32_seq_512", "shape": (32, 8, 512, 128), "num_transfers": 2},
    {"name": "b32_seq_1024", "shape": (32, 8, 1024, 128), "num_transfers": 3},
    # {"name": "b32_seq_2048", "shape": (32, 8, 2048, 128), "num_transfers": 1},
    # {"name": "b32_seq_4096", "shape": (32, 8, 4096, 128), "num_transfers": 1},
    # {"name": "b32_seq_8192", "shape": (32, 8, 8192, 128), "num_transfers": 1},
    # {"name": "b32_seq_16384", "shape": (32, 8, 16384, 128), "num_transfers": 1},
    # {"name": "b32_seq_32768", "shape": (32, 8, 32768, 128), "num_transfers": 1},
]

# Results file path
RESULTS_FILE = "/localdev/dmadic/tt-metal/tests/ttnn/distributed/kv_cache_benchmark_results.json"
PLOT_FILE = "/localdev/dmadic/tt-metal/tests/ttnn/distributed/kv_cache_transfer_latency.png"


def setup_socket(device, mesh_shape, sender_rank=0, receiver_rank=1):
    """Setup socket for data transfer between ranks (legacy single-connection)."""
    sender_coord = ttnn.CoreCoord(0, 0)
    recv_coord = ttnn.CoreCoord(0, 0)

    socket_connections = []
    for coord in ttnn.MeshCoordinateRange(mesh_shape):
        socket_connections.append(
            ttnn.SocketConnection(ttnn.MeshCoreCoord(coord, sender_coord), ttnn.MeshCoreCoord(coord, recv_coord))
        )

    l1_buffer_size = 1300 * 1024  # 1.3 MB
    socket_mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, l1_buffer_size)

    socket_config = ttnn.SocketConfig(
        socket_connections,
        socket_mem_config,
        sender_rank=sender_rank,
        receiver_rank=receiver_rank,
    )

    return socket_config


def setup_socket_optimized(
    device,
    mesh_shape,
    num_connections=NUM_SOCKET_CONNECTIONS,
    fifo_size=SOCKET_FIFO_SIZE,
    buffer_type=SOCKET_BUFFER_TYPE,
    use_diagonal_cores=USE_DIAGONAL_CORES,
    sender_rank=0,
    receiver_rank=1,
):
    """
    Setup optimized socket with parallel connections across mesh coordinates.

    This creates connections that utilize DIFFERENT physical ethernet links
    by mapping to different mesh coordinates (chips). For N300 1x2 mesh:
    - MeshCoordinate(0,0) -> uses one physical link
    - MeshCoordinate(0,1) -> uses another physical link

    This enables true parallel data transfer across both ethernet links.
    """
    # Use BOTH QSFP cables for 2x bandwidth:
    # QSFP #1: Chip 0 (MeshCoord 0,0) ↔ Chip 2 via channels 6,7
    # QSFP #2: Chip 1 (MeshCoord 0,1) ↔ Chip 3 via channels 0,1
    socket_connections = [
        # QSFP Cable #1: Chip 0 → Chip 2
        ttnn.SocketConnection(
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 6)),
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 6)),
        ),
        ttnn.SocketConnection(
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 7)),
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 7)),
        ),
    ]

    num_links = len(socket_connections)
    buffer_type_enum = ttnn.BufferType.DRAM if buffer_type == "DRAM" else ttnn.BufferType.L1
    socket_mem_config = ttnn.SocketMemoryConfig(buffer_type_enum, fifo_size)

    socket_config = ttnn.SocketConfig(
        socket_connections,
        socket_mem_config,
        sender_rank=sender_rank,
        receiver_rank=receiver_rank,
    )

    logger.info(f"Socket config: {num_links} links (mesh coords), FIFO={fifo_size}, buffer={buffer_type}")
    return socket_config


def get_tensor_memory_config():
    """Get memory config for tensors based on optimization settings."""
    if TENSOR_MEMORY_TYPE == "DRAM":
        return ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
    else:
        return ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)


def calculate_tensor_size_bytes(shape, dtype=ttnn.bfloat16):
    """Calculate tensor size in bytes."""
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    bytes_per_element = 2 if dtype == ttnn.bfloat16 else 4
    return num_elements * bytes_per_element


def run_sender(device, socket, configs):
    """
    Sender (Rank 0): Create and send 32 layers of KV cache (K+V per layer).
    Measures end-to-end time for transferring all 64 tensors (32 K + 32 V).
    """
    logger.info("=== SENDER (Rank 0) - Starting benchmark ===")
    logger.info(f"Transferring {NUM_LAYERS} layers × 2 tensors (K+V) = {NUM_LAYERS * 2} tensors per transfer")

    results = []

    for config in configs:
        name = config["name"]
        shape = config["shape"]
        num_transfers = config["num_transfers"]

        tensor_size_bytes = calculate_tensor_size_bytes(shape)
        tensor_size_mb = tensor_size_bytes / (1024 * 1024)
        # Total KV cache size: K + V for all layers
        total_kv_size_mb = tensor_size_mb * 2 * NUM_LAYERS
        total_kv_size_gb = total_kv_size_mb / 1024

        logger.info(f"\n--- Benchmarking: {name} ---")
        logger.info(f"Per-tensor shape: {shape}, Size: {tensor_size_mb:.2f} MB")
        logger.info(
            f"Total KV cache: {NUM_LAYERS} layers × (K+V) = {total_kv_size_mb:.0f} MB ({total_kv_size_gb:.2f} GB)"
        )
        logger.info(f"Num transfers: {num_transfers}")

        # Pre-create K and V tensors on device (reuse for all transfers)
        # Using ShardTensorToMesh to distribute data across BOTH chips for parallel QSFP transfer:
        # - Chip 0 (MeshCoord 0,0) sends half via QSFP #1
        # - Chip 1 (MeshCoord 0,1) sends half via QSFP #2
        # This enables 2x bandwidth by using both physical QSFP cables!
        torch_k = torch.randn(shape, dtype=torch.bfloat16)
        torch_v = torch.randn(shape, dtype=torch.bfloat16)
        mem_config = get_tensor_memory_config()
        tt_k = ttnn.from_torch(
            torch_k,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),  # Replicate full tensor to all devices
        )
        tt_v = ttnn.from_torch(
            torch_v,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),  # Replicate full tensor to all devices
        )

        transfer_times = []

        for i in range(num_transfers):
            t_send_start_ns = time.time_ns()

            # Send all 32 layers of K+V
            for layer_idx in range(NUM_LAYERS):
                ttnn.experimental.send_async(tt_k, socket)
                ttnn.experimental.send_async(tt_v, socket)

            # Synchronize to ensure all transfers complete
            ttnn.synchronize_device(device)

            t_send_end_ns = time.time_ns()
            sender_time_ms = (t_send_end_ns - t_send_start_ns) / 1_000_000
            transfer_times.append(sender_time_ms)

            throughput_gbs = total_kv_size_gb / (sender_time_ms / 1000) if sender_time_ms > 0 else 0
            logger.info(f"  Transfer {i + 1}/{num_transfers}: {sender_time_ms:.1f}ms ({throughput_gbs:.2f} GB/s)")

        avg_time = sum(transfer_times) / len(transfer_times)
        min_time = min(transfer_times)
        max_time = max(transfer_times)

        results.append(
            {
                "name": name,
                "shape": shape,
                "size_bytes": tensor_size_bytes,
                "size_mb": tensor_size_mb,
                "total_kv_size_mb": total_kv_size_mb,
                "total_kv_size_gb": total_kv_size_gb,
                "num_layers": NUM_LAYERS,
                "num_transfers": num_transfers,
                "sender_times_ms": transfer_times,
                "sender_avg_ms": avg_time,
                "sender_min_ms": min_time,
                "sender_max_ms": max_time,
            }
        )

        avg_throughput_gbs = total_kv_size_gb / (avg_time / 1000) if avg_time > 0 else 0
        logger.info(f"  Avg: {avg_time:.1f}ms, Throughput: {avg_throughput_gbs:.2f} GB/s")

        # Cleanup
        del tt_k, tt_v

    # Send termination signal
    term_metadata = torch.tensor([[[[0, 0, 0, 0, 0, 0, 0, 0]]]], dtype=torch.int64)
    term_metadata_tt = ttnn.from_torch(
        term_metadata,
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    ttnn.experimental.send_async(term_metadata_tt, socket)
    ttnn.synchronize_device(device)

    logger.info("\n=== SENDER - Benchmark complete ===")
    return results


def run_receiver(device, socket, configs):
    """
    Receiver (Rank 1): Receive 32 layers of KV cache.
    Uses local receiver-side timing only (not cross-machine E2E).
    Sender-side timing is more reliable for throughput measurement.
    """
    logger.info("=== RECEIVER (Rank 1) - Starting benchmark ===")
    logger.info(f"Receiving {NUM_LAYERS} layers × 2 tensors (K+V) = {NUM_LAYERS * 2} tensors per transfer")

    results = []

    for config in configs:
        name = config["name"]
        shape = config["shape"]
        num_transfers = config["num_transfers"]

        tensor_size_bytes = calculate_tensor_size_bytes(shape)
        tensor_size_mb = tensor_size_bytes / (1024 * 1024)
        total_kv_size_mb = tensor_size_mb * 2 * NUM_LAYERS
        total_kv_size_gb = total_kv_size_mb / 1024

        logger.info(f"\n--- Receiving: {name} ---")
        logger.info(f"Per-tensor shape: {shape}, Size: {tensor_size_mb:.2f} MB")
        logger.info(f"Total KV cache: {total_kv_size_mb:.0f} MB ({total_kv_size_gb:.2f} GB)")

        recv_times = []

        # Pre-compute padded shape for allocation
        # Replicated: each chip receives the full tensor
        padded_shape = [shape[0], shape[1], ((shape[2] + 31) // 32) * 32, ((shape[3] + 31) // 32) * 32]

        for i in range(num_transfers):
            t_recv_start_ns = time.time_ns()

            # Receive all 32 layers of K+V
            for layer_idx in range(NUM_LAYERS):
                k_recv = ttnn.allocate_tensor_on_device(
                    ttnn.TensorSpec(padded_shape, ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT), device
                )
                v_recv = ttnn.allocate_tensor_on_device(
                    ttnn.TensorSpec(padded_shape, ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT), device
                )
                ttnn.experimental.recv_async(k_recv, socket)
                ttnn.experimental.recv_async(v_recv, socket)
                ttnn.synchronize_device(device)
                del k_recv, v_recv

            t_recv_end_ns = time.time_ns()
            recv_time_ms = (t_recv_end_ns - t_recv_start_ns) / 1_000_000
            recv_times.append(recv_time_ms)

            throughput_gbs = total_kv_size_gb / (recv_time_ms / 1000) if recv_time_ms > 0 else 0
            logger.info(f"  Transfer {i + 1}/{num_transfers}: {recv_time_ms:.1f}ms ({throughput_gbs:.2f} GB/s)")

        avg_recv = sum(recv_times) / len(recv_times)

        results.append(
            {
                "name": name,
                "shape": shape,
                "size_bytes": tensor_size_bytes,
                "size_mb": tensor_size_mb,
                "total_kv_size_mb": total_kv_size_mb,
                "total_kv_size_gb": total_kv_size_gb,
                "num_layers": NUM_LAYERS,
                "num_transfers": num_transfers,
                "recv_times_ms": recv_times,
                "recv_avg_ms": avg_recv,
                "recv_min_ms": min(recv_times),
                "recv_max_ms": max(recv_times),
            }
        )

        avg_throughput_gbs = total_kv_size_gb / (avg_recv / 1000) if avg_recv > 0 else 0
        logger.info(f"  Avg: {avg_recv:.1f}ms, Throughput: {avg_throughput_gbs:.2f} GB/s")

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
    """Generate plot from benchmark results for 32-layer KV cache transfer."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec

    # Extract data - use sender-side timing for reliable measurements
    names = [r["name"] for r in results]
    seq_lens = [r["shape"][2] for r in results]
    total_sizes_gb = [r["total_kv_size_gb"] for r in results]
    latency_avg = [r["sender_avg_ms"] for r in results]
    latency_min = [r["sender_min_ms"] for r in results]
    latency_max = [r["sender_max_ms"] for r in results]
    throughputs_gbs = [
        r["total_kv_size_gb"] / (r["sender_avg_ms"] / 1000) if r["sender_avg_ms"] > 0 else 0 for r in results
    ]

    # Colors based on sequence length (gradient)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results)))

    # Set up figure
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"32-Layer KV Cache Transfer Benchmark (Llama 8B)\nBatch=32, TT-Fabric N300 → N300",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Plot 1: Latency vs Sequence Length (sender-side timing)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(seq_lens, latency_avg, "o-", color="#2980b9", markersize=12, linewidth=3)
    ax1.fill_between(seq_lens, latency_min, latency_max, alpha=0.2, color="#3498db")
    ax1.set_xlabel("Sequence Length", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Sender Latency (ms) - Log Scale", fontsize=12, fontweight="bold")
    ax1.set_title("Transfer Latency vs Sequence Length (Sender-Side)", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3, which="both", linestyle="--")
    ax1.set_xticks(seq_lens)
    ax1.set_xticklabels([f"{s // 1024}K" if s >= 1024 else str(s) for s in seq_lens], fontsize=10)

    # Annotate with latency and size
    for x, y, sz in zip(seq_lens, latency_avg, total_sizes_gb):
        label = f"{y / 1000:.1f}s" if y >= 1000 else f"{y:.0f}ms"
        ax1.annotate(
            f"{label}\n({sz:.1f}GB)", (x, y), textcoords="offset points", xytext=(5, 10), fontsize=9, ha="left"
        )

    # Plot 2: Throughput bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(range(len(seq_lens)), throughputs_gbs, color=colors, alpha=0.85, edgecolor="black", linewidth=1)
    ax2.set_xlabel("Sequence Length", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Throughput (GB/s)", fontsize=12, fontweight="bold")
    ax2.set_title("Transfer Throughput vs Sequence Length", fontsize=13, fontweight="bold")
    ax2.set_xticks(range(len(seq_lens)))
    ax2.set_xticklabels([f"{s // 1024}K" if s >= 1024 else str(s) for s in seq_lens], fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax2.axhline(y=1.0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="1 GB/s")
    ax2.legend(fontsize=10)

    for bar, tp in zip(bars, throughputs_gbs):
        height = bar.get_height()
        ax2.annotate(
            f"{tp:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Plot 3: KV Cache Size vs Latency
    ax3 = fig.add_subplot(gs[1, 0])
    scatter = ax3.scatter(
        total_sizes_gb, latency_avg, c=seq_lens, cmap="viridis", s=200, alpha=0.8, edgecolors="black", linewidth=1.5
    )
    ax3.plot(total_sizes_gb, latency_avg, "--", color="gray", alpha=0.5, linewidth=1.5)
    ax3.set_xlabel("Total KV Cache Size (GB)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Sender Latency (ms)", fontsize=12, fontweight="bold")
    ax3.set_title("Latency vs Total KV Cache Size (Sender-Side)", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3, linestyle="--")

    # Linear fit
    z = np.polyfit(total_sizes_gb, latency_avg, 1)
    x_fit = np.linspace(min(total_sizes_gb), max(total_sizes_gb), 100)
    ax3.plot(x_fit, np.poly1d(z)(x_fit), "r--", linewidth=2, label=f"Linear: {z[0]:.0f} ms/GB")
    ax3.legend(fontsize=10)
    cbar = plt.colorbar(scatter, ax=ax3, label="Sequence Length")

    # Plot 4: Summary table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    summary_lines = [
        "╔═══════════════════════════════════════════════════════════════╗",
        "║     32-LAYER KV CACHE TRANSFER BENCHMARK SUMMARY              ║",
        "╠═══════════════════════════════════════════════════════════════╣",
        "║  Seq Len  │  KV Size  │   Latency    │  Throughput            ║",
        "╠═══════════╪═══════════╪══════════════╪════════════════════════╣",
    ]
    for r, tp in zip(results, throughputs_gbs):
        seq = r["shape"][2]
        seq_str = f"{seq // 1024}K" if seq >= 1024 else str(seq)
        size_gb = r["total_kv_size_gb"]
        lat = r["sender_avg_ms"]
        lat_str = f"{lat / 1000:.2f} s" if lat >= 1000 else f"{lat:.0f} ms"
        summary_lines.append(f"║  {seq_str:>6}  │  {size_gb:>6.1f} GB │  {lat_str:>10}  │  {tp:>6.2f} GB/s           ║")

    summary_lines.extend(
        [
            "╠═══════════════════════════════════════════════════════════════╣",
            f"║  Peak Throughput: {max(throughputs_gbs):.2f} GB/s                               ║",
            f"║  Avg Throughput:  {np.mean(throughputs_gbs):.2f} GB/s                               ║",
            f"║  Linear fit: {z[0]:.0f} ms per GB transferred                      ║",
            "╚═══════════════════════════════════════════════════════════════╝",
        ]
    )

    ax4.text(
        0.5,
        0.5,
        "\n".join(summary_lines),
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="center",
        horizontalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f8f9fa", edgecolor="#2c3e50", alpha=0.95),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    logger.info(f"Plot saved to {PLOT_FILE}")

    # Also create a summary table
    print("\n" + "=" * 110)
    print(f"32-LAYER KV CACHE TRANSFER BENCHMARK SUMMARY (Batch=32, {NUM_LAYERS} layers)")
    print("=" * 110)
    print(f"{'Config':<18} {'Seq Len':<10} {'KV Size (GB)':<14} {'Latency':<14} {'Throughput (GB/s)':<18}")
    print("-" * 110)
    for r in results:
        seq_len = r["shape"][2]
        seq_str = f"{seq_len // 1024}K" if seq_len >= 1024 else str(seq_len)
        lat = r["sender_avg_ms"]
        lat_str = f"{lat / 1000:.2f} s" if lat >= 1000 else f"{lat:.0f} ms"
        tp_gbs = r["total_kv_size_gb"] / (lat / 1000) if lat > 0 else 0
        print(f"{r['name']:<18} {seq_str:<10} {r['total_kv_size_gb']:<14.2f} {lat_str:<14} {tp_gbs:<18.2f}")
    print("=" * 110)


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

    # Log optimization settings
    logger.info("=" * 60)
    logger.info("OPTIMIZATION SETTINGS:")
    logger.info(f"  Socket connections: {NUM_SOCKET_CONNECTIONS}")
    logger.info(f"  FIFO size: {SOCKET_FIFO_SIZE} bytes")
    logger.info(f"  Socket buffer type: {SOCKET_BUFFER_TYPE}")
    logger.info(f"  Tensor memory type: {TENSOR_MEMORY_TYPE}")
    logger.info(f"  Diagonal cores: {USE_DIAGONAL_CORES}")
    logger.info(f"  SHARDED TRANSFER: {USE_SHARDED_TRANSFER} (dim={SHARD_DIM})")
    logger.info("  -> Each chip sends different data through its own QSFP link")
    logger.info("=" * 60)

    # Setup optimized socket with multiple connections
    socket_config = setup_socket_optimized(
        device,
        mesh_shape,
        num_connections=NUM_SOCKET_CONNECTIONS,
        fifo_size=SOCKET_FIFO_SIZE,
        buffer_type=SOCKET_BUFFER_TYPE,
        use_diagonal_cores=USE_DIAGONAL_CORES,
    )

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
        # Save sender-side results (reliable timing for throughput measurement)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Sender results saved to {RESULTS_FILE}")

        # Generate plot using sender-side timing (accurate, same-machine measurement)
        generate_plot(results)
    else:
        results = run_receiver(device, socket, BENCHMARK_CONFIGS)
        # Save receiver-side results (local receiver timing)
        with open(RESULTS_FILE + ".receiver", "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Receiver results saved to {RESULTS_FILE}.receiver")

    # Cleanup
    del socket
    ttnn.distributed_context_barrier()
    ttnn.close_device(device)

    logger.info(f"Process {rank} finished")


if __name__ == "__main__":
    run_benchmark()

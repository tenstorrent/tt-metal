# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Chip-to-chip latency micro-benchmark on a Galaxy opened as an 8x4 ``FABRIC_2D_TORUS_XY`` mesh.

Random (src, dst) chip pairs are drawn from the full mesh and, for every tensor size, a tensor is
ping-ponged src -> dst -> src over a pair of ``ttnn.MeshSocket`` connections between 1x1 submeshes.
Each direction's ``recv_async`` gates the next ``send_async`` on the same chip, so round trips are
strictly serialized and ``elapsed / (2 * NUM_ROUND_TRIPS)`` is a true one-way latency rather than a
pipelined throughput. The loop is trace-captured, so host dispatch is excluded but per-program
launch on device is included (i.e. this is op-level latency, not raw fabric packet latency; for the
latter see ``tests/tt_metal/tt_fabric/test_infra`` ``latency_test_mode``).

``ttnn.point_to_point`` is not used: it only routes along a single mesh row/column and its sender
kernel builds a 1D packet header, which does not match the 2D torus fabric.

Env knobs:
    TT_P2P_NUM_PAIRS   number of random ordered chip pairs per tensor size (default 16)
    TT_P2P_SEED        RNG seed for pair selection (default 0); same pairs are used for every size
    TT_P2P_LATENCY_CSV output CSV path (default ./galaxy_torus_p2p_latency.csv)
"""

import csv
import os
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal

MESH_SHAPE = (8, 4)
NUM_ROUND_TRIPS = 64
NUM_TRIALS = 5
SOCKET_PAGE_SIZE = 2048  # one bf16 tile
BFLOAT16_BYTES = 2

NUM_PAIRS = int(os.environ.get("TT_P2P_NUM_PAIRS", "16"))
PAIR_SEED = int(os.environ.get("TT_P2P_SEED", "0"))

CSV_COLUMNS = [
    "tensor_shape",
    "tensor_bytes",
    "src",
    "dst",
    "row_hops",
    "col_hops",
    "torus_hops",
    "num_round_trips",
    "one_way_min_us",
    "one_way_median_us",
    "one_way_max_us",
    "effective_gbps",
]


def _torus_hops(src, dst, mesh_shape):
    """Shortest wrap-around distance along each mesh axis (dimension-ordered 2D torus routing)."""
    row = abs(src[0] - dst[0])
    col = abs(src[1] - dst[1])
    row = min(row, mesh_shape[0] - row)
    col = min(col, mesh_shape[1] - col)
    return row, col


def _random_pairs(mesh_shape, num_pairs, seed):
    coords = [(r, c) for r in range(mesh_shape[0]) for c in range(mesh_shape[1])]
    all_pairs = [(s, d) for s in coords for d in coords if s != d]
    return random.Random(seed).sample(all_pairs, min(num_pairs, len(all_pairs)))


@pytest.fixture(scope="module")
def latency_csv_writer():
    """CSV sink at ``$TT_P2P_LATENCY_CSV``; prints a size x hop-distance summary on teardown."""
    csv_path = Path(os.environ.get("TT_P2P_LATENCY_CSV", "galaxy_torus_p2p_latency.csv")).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_handle = csv_path.open("w", newline="")
    writer = csv.DictWriter(file_handle, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    print(f"\n[P2P latency] writing results to {csv_path}")

    rows: list[dict] = []

    def _append(row: dict) -> None:
        writer.writerow(row)
        file_handle.flush()
        rows.append(row)

    yield _append
    file_handle.close()

    if rows:
        print("\n" + _format_summary(rows))


def _format_summary(rows):
    """Median one-way latency (us) per tensor size (rows) and torus hop distance (columns)."""
    by_cell = defaultdict(list)
    for row in rows:
        by_cell[(row["tensor_bytes"], row["torus_hops"])].append(row["one_way_median_us"])
    sizes = sorted({k[0] for k in by_cell})
    hops = sorted({k[1] for k in by_cell})

    headers = ["tensor_bytes \\ hops"] + [str(h) for h in hops]
    table = []
    for size in sizes:
        cells = [str(size)]
        for h in hops:
            samples = by_cell.get((size, h))
            cells.append(f"{statistics.median(samples):.2f}" if samples else "-")
        table.append(cells)

    widths = [max(len(headers[i]), *(len(r[i]) for r in table)) for i in range(len(headers))]
    fmt = lambda cells: "| " + " | ".join(c.rjust(widths[i]) for i, c in enumerate(cells)) + " |"
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    title = "[P2P latency] median one-way latency (us) by tensor size and 2D-torus hop distance"
    return "\n".join([title, sep, fmt(headers), sep, *map(fmt, table), sep])


def _socket_pair(sender_mesh, receiver_mesh, sender_core, receiver_core):
    connection = ttnn.SocketConnection(
        ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), sender_core),
        ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), receiver_core),
    )
    socket_mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, SOCKET_PAGE_SIZE * 4)
    return ttnn.create_socket_pair(sender_mesh, receiver_mesh, ttnn.SocketConfig([connection], socket_mem_config))


def _measure_pair(src_mesh, dst_mesh, tensor_shape):
    """Ping-pong a tensor src -> dst -> src; returns the per-trial one-way latencies in seconds."""
    # Forward and backward sockets use disjoint cores so no core serves two sockets on one chip.
    fwd_send, fwd_recv = _socket_pair(src_mesh, dst_mesh, ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))
    bwd_send, bwd_recv = _socket_pair(dst_mesh, src_mesh, ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 1))

    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)
    src_tensor = ttnn.from_torch(
        torch_input,
        device=src_mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(src_mesh),
    )
    dst_tensor = ttnn.allocate_tensor_on_device(src_tensor.spec, dst_mesh)
    return_tensor = ttnn.allocate_tensor_on_device(src_tensor.spec, src_mesh)

    def _round_trip():
        ttnn.experimental.send_async(src_tensor, fwd_send)
        ttnn.experimental.recv_async(dst_tensor, fwd_recv)
        ttnn.experimental.send_async(dst_tensor, bwd_send)
        ttnn.experimental.recv_async(return_tensor, bwd_recv)

    # Warmup populates the program cache (required before trace capture) and checks correctness.
    _round_trip()
    ttnn.synchronize_device(src_mesh)
    ttnn.synchronize_device(dst_mesh)
    eq, msg = comp_equal(torch_input, ttnn.to_torch(dst_tensor))
    assert eq, f"forward transfer mismatch: {msg}"
    eq, msg = comp_equal(torch_input, ttnn.to_torch(return_tensor))
    assert eq, f"return transfer mismatch: {msg}"

    src_trace = ttnn.begin_trace_capture(src_mesh, cq_id=0)
    dst_trace = ttnn.begin_trace_capture(dst_mesh, cq_id=0)
    for _ in range(NUM_ROUND_TRIPS):
        _round_trip()
    ttnn.end_trace_capture(src_mesh, src_trace, cq_id=0)
    ttnn.end_trace_capture(dst_mesh, dst_trace, cq_id=0)
    ttnn.synchronize_device(src_mesh)
    ttnn.synchronize_device(dst_mesh)

    one_way_s = []
    for _ in range(NUM_TRIALS):
        start = time.perf_counter()
        ttnn.execute_trace(dst_mesh, dst_trace, cq_id=0, blocking=False)
        ttnn.execute_trace(src_mesh, src_trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(src_mesh)
        ttnn.synchronize_device(dst_mesh)
        one_way_s.append((time.perf_counter() - start) / (2 * NUM_ROUND_TRIPS))

    ttnn.release_trace(src_mesh, src_trace)
    ttnn.release_trace(dst_mesh, dst_trace)

    eq, msg = comp_equal(torch_input, ttnn.to_torch(return_tensor))
    assert eq, f"return transfer mismatch after traced replay: {msg}"

    ttnn.deallocate(src_tensor)
    ttnn.deallocate(dst_tensor)
    ttnn.deallocate(return_tensor)
    return one_way_s


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "l1_small_size": 2048,
            "trace_region_size": 8 * 1024 * 1024,
            "require_exact_physical_num_devices": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], ids=["8x4"], indirect=True)
@pytest.mark.parametrize(
    "tensor_shape",
    [(32, 32), (32, 512), (128, 1024), (512, 1024), (1024, 2048)],
    ids=lambda s: f"{s[0]}x{s[1]}",
)
def test_galaxy_torus_p2p_latency(mesh_device, tensor_shape, latency_csv_writer):
    """Measure one-way chip-to-chip latency for random chip pairs on the 8x4 2D torus."""
    if tuple(mesh_device.shape) != MESH_SHAPE:
        pytest.skip(f"needs an {MESH_SHAPE[0]}x{MESH_SHAPE[1]} Galaxy mesh, got {tuple(mesh_device.shape)}")

    submeshes = {
        (r, c): mesh_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(r, c))
        for r in range(MESH_SHAPE[0])
        for c in range(MESH_SHAPE[1])
    }
    tensor_bytes = tensor_shape[0] * tensor_shape[1] * BFLOAT16_BYTES

    for src, dst in _random_pairs(MESH_SHAPE, NUM_PAIRS, PAIR_SEED):
        one_way_s = _measure_pair(submeshes[src], submeshes[dst], tensor_shape)
        row_hops, col_hops = _torus_hops(src, dst, MESH_SHAPE)
        median_s = statistics.median(one_way_s)

        row = {
            "tensor_shape": f"{tensor_shape[0]}x{tensor_shape[1]}",
            "tensor_bytes": tensor_bytes,
            "src": f"{src[0]},{src[1]}",
            "dst": f"{dst[0]},{dst[1]}",
            "row_hops": row_hops,
            "col_hops": col_hops,
            "torus_hops": row_hops + col_hops,
            "num_round_trips": NUM_ROUND_TRIPS,
            "one_way_min_us": round(min(one_way_s) * 1e6, 3),
            "one_way_median_us": round(median_s * 1e6, 3),
            "one_way_max_us": round(max(one_way_s) * 1e6, 3),
            "effective_gbps": round(tensor_bytes / median_s / 1e9, 3),
        }
        print(
            f"[P2P latency] {row['tensor_shape']} ({tensor_bytes} B) {src}->{dst} "
            f"hops={row['torus_hops']} (row {row_hops}, col {col_hops}) | "
            f"one-way median={row['one_way_median_us']} us min={row['one_way_min_us']} us | "
            f"{row['effective_gbps']} GB/s"
        )
        latency_csv_writer(row)

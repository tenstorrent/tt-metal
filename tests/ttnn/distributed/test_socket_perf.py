# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Bandwidth micro-benchmark for ``ttnn.MeshSocket``.

The test sends a tensor between two submeshes of a single ``MeshDevice`` using
an intra-process socket pair, and measures the achieved end-to-end bandwidth.

Parameterized over:
    * Socket FIFO page size (controls the size of the L1 streaming buffer used
      by the socket runtime - has a direct impact on pipelining behavior).
    * Per-chip total tensor size in bytes (drives the number of pages
      transmitted per iteration).
    * Number of socket connections per device (1, 2 or 4). Each connection
      maps a distinct sender/receiver core pair, so more connections means
      more parallel ethernet channels per chip.

See ``tests/ttnn/distributed/test_multi_mesh.py`` for the inter-process
socket usage pattern that this benchmark mirrors at a single-process level.
"""

import csv
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal


NUM_WARMUP_ITERS = 2
NUM_MEASURED_ITERS = 10

BFLOAT16_BYTES = 2
TILE_HEIGHT = 32
TILE_WIDTH = 32

CSV_COLUMNS = [
    "transfer_mode",
    "num_connections",
    "socket_page_size_bytes",
    "per_chip_tensor_size_bytes",
    "num_chips",
    "num_iters",
    "elapsed_ms",
    "per_chip_bw_gbps",
    "aggregate_bw_gbps",
]


@pytest.fixture(scope="session")
def bandwidth_csv_writer():
    """Session-scoped CSV sink for bandwidth results.

    Output path is taken from the ``MESH_SOCKET_BW_CSV`` env var if set,
    otherwise defaults to ``mesh_socket_bandwidth.csv`` in the current
    working directory. The header is written once per session.
    """
    csv_path = Path(os.environ.get("MESH_SOCKET_BW_CSV", "mesh_socket_bandwidth.csv")).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_handle = csv_path.open("w", newline="")
    writer = csv.DictWriter(file_handle, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    file_handle.flush()
    print(f"\n[MeshSocket BW] writing results to {csv_path}")

    rows: list[dict] = []

    def _append(row: dict) -> None:
        writer.writerow(row)
        file_handle.flush()
        rows.append(row)

    yield _append
    file_handle.close()

    if rows:
        print("\n" + _format_results_table(rows))


def _format_results_table(rows: list[dict]) -> str:
    """Render the accumulated CSV rows as an ASCII table."""
    headers = CSV_COLUMNS
    str_rows = [[str(row[col]) for col in headers] for row in rows]
    widths = [max(len(headers[i]), *(len(r[i]) for r in str_rows)) for i in range(len(headers))]

    def _fmt(cells):
        return "| " + " | ".join(cell.rjust(widths[i]) for i, cell in enumerate(cells)) + " |"

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    lines = [sep, _fmt(headers), sep]
    lines.extend(_fmt(r) for r in str_rows)
    lines.append(sep)
    return "\n".join(lines)


def _build_socket_connections(mesh_shape: ttnn.MeshShape, num_connections: int):
    """Build a ``num_connections``-per-device connection list.

    Sender cores are laid out in row 0, receiver cores in row 1, so the two
    sets of cores never overlap (the socket runtime forbids a core appearing
    in two connections of the same socket).
    """
    sender_cores = [ttnn.CoreCoord(i, 0) for i in range(num_connections)]
    recv_cores = [ttnn.CoreCoord(i, 1) for i in range(num_connections)]

    connections = []
    for coord in ttnn.MeshCoordinateRange(mesh_shape):
        for sender, receiver in zip(sender_cores, recv_cores):
            connections.append(
                ttnn.SocketConnection(
                    ttnn.MeshCoreCoord(coord, sender),
                    ttnn.MeshCoreCoord(coord, receiver),
                )
            )
    return connections


def _per_chip_shape_for_bytes(total_bytes_per_chip: int):
    """Pick a tile-aligned 4D shape whose bfloat16 footprint matches ``total_bytes_per_chip``."""
    assert total_bytes_per_chip % (TILE_HEIGHT * TILE_WIDTH * BFLOAT16_BYTES) == 0, (
        f"total_bytes_per_chip={total_bytes_per_chip} must be a multiple of a bfloat16 tile "
        f"({TILE_HEIGHT * TILE_WIDTH * BFLOAT16_BYTES} bytes)"
    )
    num_elems = total_bytes_per_chip // BFLOAT16_BYTES
    # Shape it as [1, 1, TILE_HEIGHT, W] so that W is tile-aligned.
    width = num_elems // TILE_HEIGHT
    assert width % TILE_WIDTH == 0, f"derived width {width} not tile-aligned"
    return [1, 1, TILE_HEIGHT, width]


def _run_mesh_socket_bandwidth_case(
    mesh_device,
    num_connections,
    socket_page_size,
    total_tensor_size_bytes,
    bandwidth_csv_writer,
    transfer_mode="async",
) -> None:
    """Run one ``MeshSocket`` bandwidth measurement and record the result.

    ``transfer_mode`` selects between the FIFO-based ``send_async``/``recv_async`` ops and the
    direct-write ``send_direct_async``/``recv_direct_async`` ops (which bypass the socket FIFO for
    payload data and only use it for the handshake and completion signal).
    """
    torch.manual_seed(0)

    if transfer_mode == "direct":
        send_op = ttnn.experimental.send_direct_async
        recv_op = ttnn.experimental.recv_direct_async
    else:
        send_op = ttnn.experimental.send_async
        recv_op = ttnn.experimental.recv_async

    sender_mesh_device = mesh_device.create_submesh(ttnn.MeshShape(1, 2), ttnn.MeshCoordinate(0, 0))
    receiver_mesh_device = mesh_device.create_submesh(ttnn.MeshShape(1, 2), ttnn.MeshCoordinate(1, 0))

    mesh_shape = sender_mesh_device.shape
    num_chips = mesh_shape[0] * mesh_shape[1]

    socket_connections = _build_socket_connections(mesh_shape, num_connections)
    socket_mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, socket_page_size * 4)
    socket_config = ttnn.SocketConfig(socket_connections, socket_mem_config)
    send_socket, recv_socket = ttnn.create_socket_pair(sender_mesh_device, receiver_mesh_device, socket_config)

    per_chip_shape = _per_chip_shape_for_bytes(total_tensor_size_bytes)
    torch_input = torch.randn(per_chip_shape, dtype=torch.float32)
    input_tensor = ttnn.from_torch(
        torch_input,
        device=sender_mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
        mesh_mapper=ttnn.ReplicateTensorToMesh(sender_mesh_device),
    )
    output_tensor = ttnn.allocate_tensor_on_device(input_tensor.spec, receiver_mesh_device)

    for _ in range(NUM_WARMUP_ITERS):
        send_op(input_tensor, send_socket)
        recv_op(output_tensor, recv_socket)
    ttnn.synchronize_device(sender_mesh_device)
    ttnn.synchronize_device(receiver_mesh_device)

    start = time.perf_counter()
    for _ in range(NUM_MEASURED_ITERS):
        send_op(input_tensor, send_socket)
        recv_op(output_tensor, recv_socket)
    ttnn.synchronize_device(sender_mesh_device)
    ttnn.synchronize_device(receiver_mesh_device)
    elapsed_s = time.perf_counter() - start

    input_data = ttnn.to_torch(input_tensor, mesh_composer=ttnn.ConcatMeshToTensor(sender_mesh_device, dim=0))
    output_data = ttnn.to_torch(output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(receiver_mesh_device, dim=0))
    eq, msg = comp_equal(input_data, output_data)
    assert eq, msg

    bytes_per_iter_per_chip = total_tensor_size_bytes
    total_bytes = bytes_per_iter_per_chip * num_chips * NUM_MEASURED_ITERS
    per_chip_bw_gbps = (bytes_per_iter_per_chip * NUM_MEASURED_ITERS) / elapsed_s / 1e9
    aggregate_bw_gbps = total_bytes / elapsed_s / 1e9

    print(
        f"\n[MeshSocket BW] mode={transfer_mode} "
        f"num_connections={num_connections} "
        f"page_size={socket_page_size}B "
        f"per_chip_size={total_tensor_size_bytes}B "
        f"chips={num_chips} iters={NUM_MEASURED_ITERS} "
        f"elapsed={elapsed_s * 1e3:.2f}ms | "
        f"per-chip={per_chip_bw_gbps:.3f} GB/s | "
        f"aggregate={aggregate_bw_gbps:.3f} GB/s"
    )

    bandwidth_csv_writer(
        {
            "transfer_mode": transfer_mode,
            "num_connections": num_connections,
            "socket_page_size_bytes": socket_page_size,
            "per_chip_tensor_size_bytes": total_tensor_size_bytes,
            "num_chips": num_chips,
            "num_iters": NUM_MEASURED_ITERS,
            "elapsed_ms": round(elapsed_s * 1e3, 4),
            "per_chip_bw_gbps": round(per_chip_bw_gbps, 4),
            "aggregate_bw_gbps": round(aggregate_bw_gbps, 4),
        }
    )


@pytest.mark.timeout(180)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize(
    "total_tensor_size_bytes",
    [1 << 20, 1 << 24],  # 1 MiB, 16 MiB, 256 MiB per chip
    ids=lambda v: f"size{v}",
)
@pytest.mark.parametrize(
    "socket_page_size",
    [1024, 4096, 8192, 16384],
    ids=lambda v: f"page{v}",
)
@pytest.mark.parametrize(
    "num_connections",
    [2],
    ids=lambda v: f"conn{v}",
)
@pytest.mark.parametrize(
    "transfer_mode",
    ["async", "direct"],
    ids=lambda v: f"mode_{v}",
)
def test_mesh_socket_bandwidth(
    mesh_device,
    transfer_mode,
    num_connections,
    socket_page_size,
    total_tensor_size_bytes,
    bandwidth_csv_writer,
):
    """Measure ``MeshSocket`` send/recv bandwidth.

    A 2x2 mesh is split row-wise into two 1x2 submeshes; the top row is the
    sender and the bottom row is the receiver. The same per-chip tensor is
    transmitted ``NUM_MEASURED_ITERS`` times and the average bandwidth is
    reported. Correctness is verified once after the timed loop to catch
    misconfigured tests (page size too small, etc.).
    """
    _run_mesh_socket_bandwidth_case(
        mesh_device,
        num_connections,
        socket_page_size,
        total_tensor_size_bytes,
        bandwidth_csv_writer,
        transfer_mode=transfer_mode,
    )

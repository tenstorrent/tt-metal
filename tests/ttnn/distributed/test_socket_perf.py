# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Bandwidth micro-benchmark for ``ttnn.MeshSocket``.

Sends a tensor between two single-device submeshes of one ``MeshDevice`` and reports the achieved
bandwidth for both the FIFO-based ``send_async`` and the direct-write ``send_direct_async``.

Each socket connection maps a distinct sender/receiver core pair, so more connections means more
parallel channels per chip.
"""

import csv
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal


NUM_WARMUP_ITERS = 3
NUM_MEASURED_ITERS = 100

BFLOAT16_BYTES = 2

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
    """CSV sink for bandwidth results, at ``$MESH_SOCKET_BW_CSV`` or ``./mesh_socket_bandwidth.csv``."""
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
    # Senders in core row 0, receivers in row 1: the socket runtime forbids a core appearing in two
    # connections of the same socket.
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


def _run_mesh_socket_bandwidth_case(
    mesh_device,
    num_connections,
    socket_page_size,
    tensor_shape,
    bandwidth_csv_writer,
    transfer_mode="async",
) -> None:
    torch.manual_seed(0)

    if transfer_mode == "direct":
        send_op = ttnn.experimental.send_direct_async
        recv_op = ttnn.experimental.recv_direct_async
    else:
        send_op = ttnn.experimental.send_async
        recv_op = ttnn.experimental.recv_async

    sender_mesh_device = mesh_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
    receiver_mesh_device = mesh_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(1, 0))

    mesh_shape = sender_mesh_device.shape
    num_chips = mesh_shape[0] * mesh_shape[1]

    socket_connections = _build_socket_connections(mesh_shape, num_connections)
    if transfer_mode == "direct":
        # Payload bypasses the FIFO, so it only has to hold the handshake page and
        # socket_page_size does not apply.
        socket_mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 128)
    else:
        socket_mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, socket_page_size * 4)
    socket_config = ttnn.SocketConfig(socket_connections, socket_mem_config)
    send_socket, recv_socket = ttnn.create_socket_pair(sender_mesh_device, receiver_mesh_device, socket_config)

    torch_input = [torch.randn(tensor_shape, dtype=torch.bfloat16) for _ in range(NUM_WARMUP_ITERS)]
    input_tensors = [
        ttnn.from_torch(
            torch_input[i],
            device=sender_mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
            mesh_mapper=ttnn.ReplicateTensorToMesh(sender_mesh_device),
        )
        for i in range(NUM_WARMUP_ITERS)
    ]

    output_tensors = [
        ttnn.allocate_tensor_on_device(input_tensors[i].spec, receiver_mesh_device) for i in range(NUM_WARMUP_ITERS)
    ]

    # Warmup doubles as the correctness check and populates the program cache, which trace capture
    # below requires: building a program mid-capture would fail.
    for i in range(NUM_WARMUP_ITERS):
        send_op(input_tensors[i], send_socket)
        recv_op(output_tensors[i], recv_socket)
    for i in range(NUM_WARMUP_ITERS):
        output_data = ttnn.to_torch(output_tensors[i])
        eq, msg = comp_equal(torch_input[i], output_data)
        assert eq, f"warmup iteration {i}: {msg}"

    sender_trace = ttnn.begin_trace_capture(sender_mesh_device, cq_id=0)
    receiver_trace = ttnn.begin_trace_capture(receiver_mesh_device, cq_id=0)
    for _ in range(NUM_MEASURED_ITERS):
        send_op(input_tensors[0], send_socket)
        recv_op(output_tensors[0], recv_socket)
    ttnn.end_trace_capture(sender_mesh_device, sender_trace, cq_id=0)
    ttnn.end_trace_capture(receiver_mesh_device, receiver_trace, cq_id=0)
    ttnn.synchronize_device(sender_mesh_device)
    ttnn.synchronize_device(receiver_mesh_device)
    start = time.perf_counter()
    ttnn.execute_trace(sender_mesh_device, sender_trace, cq_id=0, blocking=False)
    ttnn.execute_trace(receiver_mesh_device, receiver_trace, cq_id=0, blocking=True)
    ttnn.synchronize_device(sender_mesh_device)
    ttnn.synchronize_device(receiver_mesh_device)
    elapsed_s = time.perf_counter() - start

    bytes_per_iter_per_chip = tensor_shape[0] * tensor_shape[1] * BFLOAT16_BYTES
    total_bytes = bytes_per_iter_per_chip * num_chips * NUM_MEASURED_ITERS
    per_chip_bw_gbps = (bytes_per_iter_per_chip * NUM_MEASURED_ITERS) / elapsed_s / 1e9
    aggregate_bw_gbps = total_bytes / elapsed_s / 1e9

    print(
        f"\n[MeshSocket BW] mode={transfer_mode} "
        f"num_connections={num_connections} "
        f"page_size={socket_page_size}B "
        f"per_chip_size={bytes_per_iter_per_chip}B "
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
            "per_chip_tensor_size_bytes": bytes_per_iter_per_chip,
            "num_chips": num_chips,
            "num_iters": NUM_MEASURED_ITERS,
            "elapsed_ms": round(elapsed_s * 1e3, 4),
            "per_chip_bw_gbps": round(per_chip_bw_gbps, 4),
            "aggregate_bw_gbps": round(aggregate_bw_gbps, 4),
        }
    )


fabric_router_config = ttnn.FabricRouterConfig()
fabric_router_config.max_packet_payload_size_bytes = 1088 * 8


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "l1_small_size": 2048,
            "fabric_router_config": fabric_router_config,
            "require_exact_physical_num_devices": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 4), (8, 4)], ids=["2x4", "8x4"], indirect=True)
@pytest.mark.parametrize(
    "tensor_shape",
    [[1024, 2048]],
    ids=lambda v: f"size{v}",
)
@pytest.mark.parametrize(
    "socket_page_size",
    [2048],
    ids=lambda v: f"page{v}",
)
@pytest.mark.parametrize(
    "num_connections",
    [1, 2],
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
    tensor_shape,
    bandwidth_csv_writer,
):
    """Measure ``MeshSocket`` send/recv bandwidth from the device at (0, 0) to the one at (1, 0).

    The tensor is transmitted ``NUM_MEASURED_ITERS`` times from a captured trace and the average
    bandwidth is reported. Correctness is checked during warmup, before the timed replay.
    """
    _run_mesh_socket_bandwidth_case(
        mesh_device,
        num_connections,
        socket_page_size,
        tensor_shape,
        bandwidth_csv_writer,
        transfer_mode=transfer_mode,
    )

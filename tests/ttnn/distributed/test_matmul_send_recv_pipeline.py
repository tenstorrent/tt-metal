# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Host-side timing comparison for a two-matmul pipeline.

The pipeline is:
    [32, 4096] @ [4096, 4096] -> [32, 4096]
    [32, 4096] @ [4096, 512]  -> [32, 512]

It is measured four ways:
    * both matmuls on the same 1x1 submesh
    * first matmul on one 1x1 submesh, second matmul on another, with send_async/recv_async
    * same split with send_direct_async/recv_direct_async
    * same split with buffered_send/buffered_recv
"""

import time

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


NUM_WARMUP_ITERS = 1
NUM_MEASURED_ITERS = 10
SOCKET_PAGE_SIZE = 2048
NUM_SOCKET_CONNECTIONS = 1
NUM_BUFFERED_RECV_BUFFERS = 3

M = 32
K = 4096
N0 = 512
N1 = 4096


def _build_socket_connections(mesh_shape: ttnn.MeshShape, num_connections: int):
    sender_cores = [ttnn.CoreCoord(i, 0) for i in range(num_connections)]
    recv_cores = [ttnn.CoreCoord(i, 1) for i in range(num_connections)]

    connections = []
    for coord in ttnn.MeshCoordinateRange(mesh_shape):
        for sender_core, receiver_core in zip(sender_cores, recv_cores):
            connections.append(
                ttnn.SocketConnection(
                    ttnn.MeshCoreCoord(coord, sender_core),
                    ttnn.MeshCoreCoord(coord, receiver_core),
                )
            )
    return connections


def _make_socket_pair(sender_mesh_device, receiver_mesh_device):
    socket_connections = _build_socket_connections(sender_mesh_device.shape, NUM_SOCKET_CONNECTIONS)
    socket_mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, SOCKET_PAGE_SIZE * 4)
    socket_config = ttnn.SocketConfig(socket_connections, socket_mem_config)
    return ttnn.create_socket_pair(sender_mesh_device, receiver_mesh_device, socket_config)


def _to_tt_tensor(torch_tensor, mesh_device, memory_config):
    return ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _first_matmul(input_tensor, weight_tensor):
    return ttnn.matmul(input_tensor, weight_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)


def _second_matmul(hidden_tensor, weight_tensor):
    return ttnn.matmul(hidden_tensor, weight_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)


def _sync_devices(*mesh_devices):
    for mesh_device in mesh_devices:
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)


def _time_pipeline(name, run_once, sync_fn):
    output = None
    for _ in range(NUM_WARMUP_ITERS):
        output = run_once()
    sync_fn()

    start = time.perf_counter()
    for _ in range(NUM_MEASURED_ITERS):
        output = run_once()
    sync_fn()
    elapsed_s = time.perf_counter() - start

    row = {
        "mode": name,
        "iters": NUM_MEASURED_ITERS,
        "elapsed_ms": elapsed_s * 1e3,
        "avg_ms": elapsed_s * 1e3 / NUM_MEASURED_ITERS,
    }
    return row, output


def _format_timing_table(rows):
    headers = ["mode", "iters", "elapsed_ms", "avg_ms"]
    str_rows = [
        [
            row["mode"],
            str(row["iters"]),
            f"{row['elapsed_ms']:.3f}",
            f"{row['avg_ms']:.3f}",
        ]
        for row in rows
    ]
    widths = [max(len(headers[i]), *(len(row[i]) for row in str_rows)) for i in range(len(headers))]

    def fmt(cells):
        return "| " + " | ".join(cells[i].rjust(widths[i]) for i in range(len(cells))) + " |"

    sep = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    lines = [sep, fmt(headers), sep]
    lines.extend(fmt(row) for row in str_rows)
    lines.append(sep)
    return "\n".join(lines)


def _to_host(output_tensor, mesh_device):
    return ttnn.to_torch(output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D, "l1_small_size": 2048}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_two_matmul_pipeline_transfer_host_time(mesh_device):
    torch.manual_seed(0)

    sender_mesh_device = mesh_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
    same_mesh_device = sender_mesh_device
    receiver_mesh_device = mesh_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 1))

    torch_input = torch.randn((M, K), dtype=torch.bfloat16)
    torch_weight0 = torch.randn((K, N0), dtype=torch.bfloat16)
    torch_weight1 = torch.randn((N0, N1), dtype=torch.bfloat16)

    input_same = _to_tt_tensor(torch_input, same_mesh_device, ttnn.L1_MEMORY_CONFIG)
    weight0_same = _to_tt_tensor(torch_weight0, same_mesh_device, ttnn.DRAM_MEMORY_CONFIG)
    weight1_same = _to_tt_tensor(torch_weight1, same_mesh_device, ttnn.DRAM_MEMORY_CONFIG)

    input_sender = _to_tt_tensor(torch_input, sender_mesh_device, ttnn.L1_MEMORY_CONFIG)
    weight0_sender = _to_tt_tensor(torch_weight0, sender_mesh_device, ttnn.DRAM_MEMORY_CONFIG)
    weight1_receiver = _to_tt_tensor(torch_weight1, receiver_mesh_device, ttnn.DRAM_MEMORY_CONFIG)

    rows = []
    outputs = {}

    def run_same_device_once():
        hidden = _first_matmul(input_same, weight0_same)
        return _second_matmul(hidden, weight1_same)

    row, outputs["same_device"] = _time_pipeline(
        "same_device",
        run_same_device_once,
        lambda: _sync_devices(same_mesh_device),
    )
    rows.append(row)

    def run_split_mode(transfer_mode):
        send_socket, recv_socket = _make_socket_pair(sender_mesh_device, receiver_mesh_device)
        transfer_output = None
        buffered_outputs = None
        transfer_count = 0

        def run_once():
            nonlocal transfer_output, buffered_outputs, transfer_count

            hidden_sender = _first_matmul(input_sender, weight0_sender)
            if transfer_mode == "async":
                if transfer_output is None:
                    transfer_output = ttnn.allocate_tensor_on_device(hidden_sender.spec, receiver_mesh_device)
                ttnn.experimental.send_async(hidden_sender, send_socket)
                ttnn.experimental.recv_async(transfer_output, recv_socket)
                hidden_receiver = transfer_output
            elif transfer_mode == "direct":
                if transfer_output is None:
                    transfer_output = ttnn.allocate_tensor_on_device(hidden_sender.spec, receiver_mesh_device)
                ttnn.experimental.send_direct_async(hidden_sender, send_socket)
                ttnn.experimental.recv_direct_async(transfer_output, recv_socket)
                hidden_receiver = transfer_output
            elif transfer_mode == "buffered":
                if buffered_outputs is None:
                    buffered_outputs = [
                        ttnn.allocate_tensor_on_device(hidden_sender.spec, receiver_mesh_device)
                        for _ in range(NUM_BUFFERED_RECV_BUFFERS)
                    ]
                buffer_idx = transfer_count % NUM_BUFFERED_RECV_BUFFERS
                ttnn.experimental.buffered_send(hidden_sender, send_socket)
                ttnn.experimental.buffered_recv(buffered_outputs, recv_socket)
                hidden_receiver = buffered_outputs[buffer_idx]
            else:
                raise ValueError(f"Unsupported transfer mode: {transfer_mode}")

            transfer_count += 1
            return _second_matmul(hidden_receiver, weight1_receiver)

        return _time_pipeline(
            transfer_mode,
            run_once,
            lambda: _sync_devices(sender_mesh_device, receiver_mesh_device),
        )

    for transfer_mode in ["async", "direct", "buffered"]:
        row, outputs[transfer_mode] = run_split_mode(transfer_mode)
        rows.append(row)

    print("\n[Two Matmul Pipeline Host Time]\n" + _format_timing_table(rows))

    expected = _to_host(outputs["same_device"], same_mesh_device)
    for transfer_mode in ["async", "direct", "buffered"]:
        actual = _to_host(outputs[transfer_mode], receiver_mesh_device)
        (
            passing,
            message,
        ) = comp_pcc(expected, actual, pcc=0.99)
        assert passing, f"{transfer_mode} output mismatch vs same_device: {message}"

#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MPISocket transfer repro: 100 replicated tensors, 4 devices -> 4 devices.

  * One MPISocket (SocketType.MPI), opened once and reused for every tensor.
  * Both ranks open a [1, 4] mesh; each tensor is REPLICATED across all 4 devices on both
    sides. Transport is host-staged MPI, addressed by rank.
  * Genuine 4 -> 4 transfer: a single sock.send(tensor) on a [1, 4]-replicated tensor emits
    one MPI message per device shard (4 total), and sock.recv(template) fills all 4 device
    shards of the receiver template. No 1->1 + broadcast, no per-device SocketConnection
    routing (MPISocket.send/recv move the whole tensor's shards by rank).
  * The receiver issues ALL receives first and only calls to_torch AFTER the last receive --
    no host read is interleaved between recv calls.

MPISocket still builds a MeshSocket internally, which requires a non-empty connection config,
so we hand it one trivial (0, 0) -> (0, 0) connection as construction boilerplate (unused for
MPI data routing).

Requires the ttnn build that exposes ttnn._ttnn.multi_device.create_socket / SocketType (the
binding in ttnn/core/distributed/distributed_nanobind.cpp). Rebuild _ttnn if the import fails.

Knobs:
  REPRO_NUM_TENSORS   tensors streamed (env, default 100).
  REPRO_TENSOR_SHAPE  per-device shape (env CSV, default "1,1,32,32").
  REPRO_BRANCH        git branch label for logging (env, normally set by runner.sh).

Run: bash debug_mpi_socket/runner.sh   (from tt-train/sources/examples/grpo)
"""

import os
import subprocess
import time

import torch
from ttnn._ttnn.multi_device import SocketType, create_socket
from ttnn._ttnn.multi_device import recv_bytes as _recv
from ttnn._ttnn.multi_device import send_bytes as _send

import ttnn

SENDER_RANK = 0
RECEIVER_RANK = 1
NUM_DEVICES = 4  # both ranks open a [1, 4] mesh; every tensor is replicated across all 4
NUM_TENSORS = int(os.environ.get("REPRO_NUM_TENSORS", "100"))
TENSOR_SHAPE = [int(d) for d in os.environ.get("REPRO_TENSOR_SHAPE", "1,1,32,32").split(",")]
SOCKET_FIFO_BYTES = 80 * 1024 * 1024


def _git_branch():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


BRANCH = os.environ.get("REPRO_BRANCH") or _git_branch()
RANK = os.environ.get("OMPI_COMM_WORLD_RANK", "?")


def log(msg):
    print(f"[mpi-repro rank {RANK} branch={BRANCH}] {msg}", flush=True)


def log_progress(action, tensor_index):
    step = max(1, NUM_TENSORS // 10)
    if tensor_index % step == 0 or tensor_index == NUM_TENSORS - 1:
        log(f"{action} {tensor_index + 1}/{NUM_TENSORS}")


def tensor_value(tensor_index):
    """Fill value for tensor `tensor_index`.

    Distinct per tensor so a misroute or reordering is caught, and small enough that bf16
    holds it exactly (keep NUM_TENSORS <= 256). Replicated, so every device holds this value.
    """
    return float(tensor_index + 1)


def tensor_stats(t):
    """(min, max, mean) of a torch tensor, computed in float so bf16 is summarised exactly."""
    f = t.float()
    return f.min().item(), f.max().item(), f.mean().item()


def count_correct(got, expected):
    """How many elements of `got` exactly equal `expected` (a same-shape tensor), and the total."""
    correct = int(torch.eq(got, expected).sum().item())
    return correct, got.numel()


def handshake(role):
    """Two-rank rendezvous before opening the socket: eager send, then blocking recv.

    Keeps both ranks in lockstep so the MeshSocket descriptor exchange inside create_socket
    does not trip its handshake timeout if one side is slow to reach the open.
    """
    peer_rank, send_tag, recv_tag = (RECEIVER_RANK, 7, 8) if role == "sender" else (SENDER_RANK, 8, 7)
    _send(b"r", peer_rank, send_tag)
    _recv(1, peer_rank, recv_tag)


def init():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()


def one_connection_config():
    """A SocketConfig with a single (0, 0) -> (0, 0) connection.

    MPISocket wraps a MeshSocket, whose constructor requires a non-empty connection config.
    The connection is construction boilerplate only -- MPI routes the tensor by rank, not by
    this connection.
    """
    memory = ttnn.SocketMemoryConfig(ttnn.BufferType.DRAM, SOCKET_FIFO_BYTES)
    coord = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0))
    connection = ttnn.SocketConnection(coord, coord)
    return ttnn.SocketConfig([connection], memory, sender_rank=SENDER_RANK, receiver_rank=RECEIVER_RANK)


def make_replicated_tensor(mesh, fill):
    """A [1, 4]-replicated tensor: the same TENSOR_SHAPE buffer `fill` on every device."""
    host = torch.full(TENSOR_SHAPE, fill).to(torch.bfloat16)
    return ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def open_mesh():
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, NUM_DEVICES), offset=ttnn.MeshCoordinate(0, 0))


def banner():
    log(f"START shape={TENSOR_SHAPE} num_tensors={NUM_TENSORS} num_devices={NUM_DEVICES} transport=MPI")


def sender():
    init()
    banner()
    mesh = open_mesh()

    handshake("sender")
    sock = create_socket(SocketType.MPI, mesh, RECEIVER_RANK, one_connection_config())
    log("socket open")

    for j in range(NUM_TENSORS):
        tensor = make_replicated_tensor(mesh, tensor_value(j))
        sock.send(tensor)  # one call -> all 4 replicated device copies, host-staged over MPI
        log_progress("send", j)

    log("synchronizing sender ...")
    ttnn.synchronize_device(mesh)
    log("sender done")
    handshake("sender")
    ttnn.close_mesh_device(mesh)


def receiver():
    init()
    banner()
    mesh = open_mesh()

    handshake("receiver")
    sock = create_socket(SocketType.MPI, mesh, SENDER_RANK, one_connection_config())
    log("socket open")

    # Pre-allocate all receive templates and drain their zero-fills, so the timer below measures
    # only the receives (not template allocation).
    templates = [make_replicated_tensor(mesh, 0.0) for _ in range(NUM_TENSORS)]
    ttnn.synchronize_device(mesh)

    # Phase 1: receive every tensor (each recv fills all 4 device shards); no to_torch here.
    t0 = time.perf_counter()
    for j in range(NUM_TENSORS):
        sock.recv(templates[j])  # blocking; fills all 4 replicated device copies
        log_progress("recv", j)
    ttnn.synchronize_device(mesh)  # after all recv
    recv_secs = time.perf_counter() - t0
    log(f"all {NUM_TENSORS} receives + sync: {recv_secs:.4f}s ({recv_secs / NUM_TENSORS * 1e3:.3f} ms/tensor)")
    recvd = templates

    # Phase 2: only after ALL receives, read the tensors back with to_torch and verify.
    log("verifying ...")
    all_correct = True
    for j in range(NUM_TENSORS):
        expected = torch.full(TENSOR_SHAPE, tensor_value(j), dtype=torch.bfloat16)
        for i in range(NUM_DEVICES):
            got = ttnn.to_torch(ttnn.get_device_tensors(recvd[j])[i])
            mn, mx, me = tensor_stats(got)
            correct, total = count_correct(got, expected)
            status = "OK" if correct == total else "CORRUPT"
            log(
                f"[recv] tensor {j} dev {i}: min={mn:g} max={mx:g} mean={me:g} "
                f"expected={tensor_value(j):g} correct={correct}/{total} {status}"
            )
            if correct != total:
                all_correct = False

    log(f"all {NUM_DEVICES * NUM_TENSORS} shards correct: {all_correct}")
    handshake("receiver")
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    if int(os.environ["OMPI_COMM_WORLD_RANK"]) == SENDER_RANK:
        sender()
    else:
        receiver()

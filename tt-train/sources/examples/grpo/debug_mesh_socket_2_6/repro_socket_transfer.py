#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MeshSocket (fabric) 2->6 repro: 1 device -> 1 device then replicate -- FATALs on routing.

Same 1->1-transfer-then-broadcast logic as debug_mesh_socket_and_broadcast, but with an
ASYMMETRIC split: the sender opens a [1, 2] mesh (1 N300 board) and the receiver a [1, 6] mesh
(3 boards). A single MeshSocket connection sender(0, 0) -> receiver(0, 0) carries device 0's
shard; the receiver then broadcasts it across its [1, 6] mesh.

!!! THIS IS EXPECTED TO FATAL AT open_mesh_device -- IT DOES NOT COMPLETE A TRANSFER !!!
A single-board sender mesh is an asymmetric inter-mesh endpoint. On a T3000 (4x N300, 2x4
mesh) that board's local chip has ethernet cables to two other boards, so fabric inter-mesh
bring-up trips:
    control_plane.cpp: "Inter-mesh: one src to multiple dst chips ... not supported yet"
That check runs during open_mesh_device, BEFORE any socket -- so the run dies before the
socket/transfer code is reached. This folder exists to document/reproduce that routing
failure; the working symmetric version is debug_mesh_socket_and_broadcast (4/4).

Knobs:
  REPRO_NUM_TENSORS   tensors streamed (env, default 100).
  REPRO_TENSOR_SHAPE  per-device shape (env CSV, default "1,1,32,32").
  REPRO_BRANCH        git branch label for logging (env, normally set by runner.sh).

Run: bash debug_mesh_socket_2_6/runner.sh   (from tt-train/sources/examples/grpo)
"""

import os
import subprocess
import time

import torch
from ttnn._ttnn.multi_device import recv_bytes as _recv
from ttnn._ttnn.multi_device import send_bytes as _send

import ttnn

SENDER_RANK = 0
RECEIVER_RANK = 1
SENDER_DEVICES = 2  # sender opens a [1, 2] mesh (1 board); only coordinate (0, 0) is transferred
RECEIVER_DEVICES = 6  # receiver opens a [1, 6] mesh (3 boards) and replicates the shard across all 6
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
    print(f"[mesh-repro rank {RANK} branch={BRANCH}] {msg}", flush=True)


def log_progress(action, tensor_index):
    step = max(1, NUM_TENSORS // 10)
    if tensor_index % step == 0 or tensor_index == NUM_TENSORS - 1:
        log(f"{action} {tensor_index + 1}/{NUM_TENSORS}")


def tensor_value(tensor_index):
    """Fill value for tensor `tensor_index`.

    Distinct per tensor so a misroute or reordering is caught, and small enough that bf16
    holds it exactly (keep NUM_TENSORS <= 256).
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

    Keeps both ranks in lockstep so the MeshSocket descriptor exchange does not trip its
    handshake timeout if one side is slow to reach the open.
    """
    peer_rank, send_tag, recv_tag = (RECEIVER_RANK, 7, 8) if role == "sender" else (SENDER_RANK, 8, 7)
    _send(b"r", peer_rank, send_tag)
    _recv(1, peer_rank, recv_tag)


def init():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()


def single_connection_config():
    """A SocketConfig with ONE connection: sender (0, 0) -> receiver (0, 0).

    This is the whole point of this repro: a single device-to-device link, no fan-out.
    """
    memory = ttnn.SocketMemoryConfig(ttnn.BufferType.DRAM, SOCKET_FIFO_BYTES)
    coord = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0))
    connection = ttnn.SocketConnection(coord, coord)
    return ttnn.SocketConfig([connection], memory, sender_rank=SENDER_RANK, receiver_rank=RECEIVER_RANK)


def open_mesh(num_devices):
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, num_devices), offset=ttnn.MeshCoordinate(0, 0))


def make_replicated_tensor(mesh, fill):
    """A tensor holding `fill`, replicated across every device of the given mesh.

    On the sender ([1, 2]), only coordinate (0, 0)'s copy is transferred over the
    single-connection socket. On the receiver ([1, 6]), this allocates the recv template (and,
    after fan-out, the final replicated result across all 6 devices).
    """
    host = torch.full(TENSOR_SHAPE, fill).to(torch.bfloat16)
    return ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def banner():
    log(
        f"START shape={TENSOR_SHAPE} num_tensors={NUM_TENSORS} "
        f"sender_devices={SENDER_DEVICES} receiver_devices={RECEIVER_DEVICES} transport=MeshSocket(fabric)"
    )


def sender():
    init()
    banner()
    mesh = open_mesh(SENDER_DEVICES)

    handshake("sender")
    socket = ttnn.MeshSocket(mesh, single_connection_config())
    log("socket open (1 connection: (0,0) -> (0,0))")

    for j in range(NUM_TENSORS):
        # Only coordinate (0, 0)'s copy crosses the wire (single connection).
        tensor = make_replicated_tensor(mesh, tensor_value(j))
        ttnn.experimental.send_async(tensor, socket)
        log_progress("send", j)

    log("synchronizing sender ...")
    ttnn.synchronize_device(mesh)
    log("sender done")
    handshake("sender")
    ttnn.close_mesh_device(mesh)


def receiver():
    init()
    banner()
    mesh = open_mesh(RECEIVER_DEVICES)

    handshake("receiver")
    socket = ttnn.MeshSocket(mesh, single_connection_config())
    log("socket open (1 connection: (0,0) -> (0,0))")

    # Pre-allocate all receive templates and drain their zero-fills, so the timer below measures
    # only the receives. Each recv_async writes the single shard onto coordinate (0, 0) of its
    # template; the broadcast (later) fans it out across the [1, RECEIVER_DEVICES] mesh.
    templates = [make_replicated_tensor(mesh, 0.0) for _ in range(NUM_TENSORS)]
    ttnn.synchronize_device(mesh)

    # Phase 1: issue every recv_async back-to-back (no to_torch, no broadcast in between), then a
    # single synchronize_device to complete them all. Time the whole receive phase.
    t0 = time.perf_counter()
    for j in range(NUM_TENSORS):
        ttnn.experimental.recv_async(templates[j], socket)
        log_progress("recv issued", j)
    ttnn.synchronize_device(mesh)  # after all recv: force every recv_async to complete
    recv_secs = time.perf_counter() - t0
    log(f"all {NUM_TENSORS} receives + sync: {recv_secs:.4f}s ({recv_secs / NUM_TENSORS * 1e3:.3f} ms/tensor)")

    # Replicate on the receiving side (after all recvs complete): broadcast coordinate (0, 0)'s
    # shard out to every device of the [1, RECEIVER_DEVICES] mesh.
    recvd = [ttnn.broadcast(templates[j], ttnn.MeshCoordinate(0, 0)) for j in range(NUM_TENSORS)]

    # Phase 2: read the tensors back with to_torch and verify.
    log("verifying ...")
    all_correct = True
    for j in range(NUM_TENSORS):
        expected = torch.full(TENSOR_SHAPE, tensor_value(j), dtype=torch.bfloat16)
        for i in range(RECEIVER_DEVICES):
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

    log(f"all {RECEIVER_DEVICES * NUM_TENSORS} replicated shards correct: {all_correct}")
    handshake("receiver")
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    if int(os.environ["OMPI_COMM_WORLD_RANK"]) == SENDER_RANK:
        sender()
    else:
        receiver()

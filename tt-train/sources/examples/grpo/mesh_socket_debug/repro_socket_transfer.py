#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Configurable multi-tensor MeshSocket transfer repro with three transfer modes.

A sender rank streams NUM_TENSORS sharded tensors to a receiver rank over MeshSockets and the
receiver verifies every shard element-by-element. Two independent knobs select the topology so
we can tell whether a hang/corruption is driven by submeshes or by the number of sockets:

  USE_SUBMESH  SINGLE_SOCKET  mode           receiver topology / sockets
  0            1              bigmesh_1sock  one [1, N] mesh; ONE socket, N conns (0, i) -> (0, i)
  0            0              bigmesh_Nsock  one [1, N] mesh; N sockets, conn (0, i) -> (0, i)
  1            0              submesh_Nsock  N x [1, 1] submeshes; N sockets, conn (0, i) -> (0, 0)

USE_SUBMESH=1 + SINGLE_SOCKET=1 is impossible: a MeshSocket binds to exactly one mesh
(tt_metal/distributed/mesh_socket.cpp asserts local_mesh_binding.size() == 1), so one socket
cannot span N separate [1, 1] submeshes -- the submesh path forces one-socket-per-submesh.

Each tensor's shard on device i is filled with a distinct constant shard_value(i, j). The sender
prints min/max/mean of every shard before sending; the receiver prints min/max/mean plus an exact
correct/total element count for every (tensor, device) -- so it is obvious whether a shard is
correct, and if not, exactly how corrupted it is and where.

Knobs:
  NUM_DEVICES          devices per side = N. Must match the runner's mesh config (local8 -> 4).
  NUM_TENSORS          tensors streamed per socket (env REPRO_NUM_TENSORS, default 100).
  TENSOR_SHAPE         per-device shard shape (env REPRO_TENSOR_SHAPE, e.g. "1,1,8192,4096").
  REPRO_USE_SUBMESH    receiver splits into [1, 1] submeshes (env 0/1, default 0).
  REPRO_SINGLE_SOCKET  one socket with N connections instead of N sockets (env 0/1, default 0).
  REPRO_BRANCH         git branch label for logging (env, normally set by runner.sh).

Run: bash mesh_socket_debug/runner.sh   (from tt-train/sources/examples/grpo)
"""

import os
import subprocess

import torch
from ttnn._ttnn.multi_device import recv_bytes as _recv
from ttnn._ttnn.multi_device import send_bytes as _send

import ttnn

SENDER_RANK = 0
RECEIVER_RANK = 1
NUM_DEVICES = 4
NUM_TENSORS = int(os.environ.get("REPRO_NUM_TENSORS", "100"))
TENSOR_SHAPE = [int(d) for d in os.environ.get("REPRO_TENSOR_SHAPE", "1,1,32,32").split(",")]
USE_SUBMESH = os.environ.get("REPRO_USE_SUBMESH", "0") == "1"
SINGLE_SOCKET = os.environ.get("REPRO_SINGLE_SOCKET", "0") == "1"
SOCKET_FIFO_BYTES = 80 * 1024 * 1024

if USE_SUBMESH and SINGLE_SOCKET:
    raise SystemExit(
        "REPRO_USE_SUBMESH=1 + REPRO_SINGLE_SOCKET=1 is impossible: a MeshSocket binds to "
        "exactly one mesh, so one socket cannot span N separate [1, 1] submeshes."
    )

MODE = f"{'submesh' if USE_SUBMESH else 'bigmesh'}_{'1sock' if SINGLE_SOCKET else 'Nsock'}"


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
    print(f"[repro rank {RANK} branch={BRANCH} mode={MODE}] {msg}", flush=True)


def log_progress(action, tensor_index):
    step = max(1, NUM_TENSORS // 10)
    if tensor_index % step == 0 or tensor_index == NUM_TENSORS - 1:
        log(f"{action} {tensor_index + 1}/{NUM_TENSORS}")


def shard_value(device_index, tensor_index):
    """Fill value for the shard on device `device_index` of tensor `tensor_index`.

    Distinct per (device, tensor) so a misroute or reordering is caught, and small enough that
    bf16 holds it exactly (keep NUM_DEVICES + NUM_TENSORS <= 256).
    """
    return float(device_index + 1 + tensor_index)


def tensor_stats(t):
    """(min, max, mean) of a torch tensor, computed in float so bf16 is summarised exactly."""
    f = t.float()
    return f.min().item(), f.max().item(), f.mean().item()


def count_correct(got, expected):
    """How many elements of `got` exactly equal `expected` (a same-shape tensor), and the total."""
    correct = int(torch.eq(got, expected).sum().item())
    return correct, got.numel()


def handshake(role):
    """Two-rank rendezvous before each socket open: eager send, then blocking recv."""
    peer_rank, send_tag, recv_tag = (RECEIVER_RANK, 7, 8) if role == "sender" else (SENDER_RANK, 8, 7)
    _send(b"r", peer_rank, send_tag)
    _recv(1, peer_rank, recv_tag)


def _socket_config(connections):
    memory = ttnn.SocketMemoryConfig(ttnn.BufferType.DRAM, SOCKET_FIFO_BYTES)
    return ttnn.SocketConfig(connections, memory, sender_rank=SENDER_RANK, receiver_rank=RECEIVER_RANK)


def bigmesh_single_config():
    """ONE socket carrying N connections (0, i) -> (0, i) on the full [1, N] mesh."""
    connections = [
        ttnn.SocketConnection(
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, i), ttnn.CoreCoord(0, 0)),
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, i), ttnn.CoreCoord(0, 0)),
        )
        for i in range(NUM_DEVICES)
    ]
    return _socket_config(connections)


def bigmesh_socket_config(socket_index):
    """One socket carrying a single connection (0, i) -> (0, i) on the full [1, N] mesh."""
    coord = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, socket_index), ttnn.CoreCoord(0, 0))
    return _socket_config([ttnn.SocketConnection(coord, coord)])


def submesh_socket_config(socket_index):
    """One socket: sender (0, i) on the [1, N] mesh -> receiver (0, 0) on the i-th [1, 1] submesh."""
    sender = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, socket_index), ttnn.CoreCoord(0, 0))
    receiver = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0))
    return _socket_config([ttnn.SocketConnection(sender, receiver)])


def init():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()


def make_sharded_tensor(mesh, tensor_index):
    """One [N, *] tensor whose shard on device i is filled with shard_value(i, tensor_index)."""
    shards = [torch.full(TENSOR_SHAPE, shard_value(i, tensor_index)) for i in range(NUM_DEVICES)]
    host = torch.cat(shards, dim=0).to(torch.bfloat16)
    return ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, 0),
    )


def sharded_recv_spec(mesh):
    """Spec of a [N, *] tensor sharded dim-0 across the big mesh -- one shard (TENSOR_SHAPE) per
    device, matching make_sharded_tensor. Used for the bigmesh recv templates."""
    full_shape = [NUM_DEVICES, *TENSOR_SHAPE[1:]]
    ref = ttnn.from_torch(
        torch.zeros(full_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, 0),
    )
    return ref.spec


def submesh_recv_spec(submesh):
    """Spec of a single-device [TENSOR_SHAPE] tensor on a [1, 1] submesh."""
    ref = ttnn.from_torch(
        torch.zeros(TENSOR_SHAPE, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    return ref.spec


def banner():
    log(
        f"START shape={TENSOR_SHAPE} num_tensors={NUM_TENSORS} num_devices={NUM_DEVICES} "
        f"use_submesh={int(USE_SUBMESH)} single_socket={int(SINGLE_SOCKET)}"
    )


def open_sender_sockets(mesh):
    """Open the sender-side sockets (one handshake per open; same count/order as the receiver)."""
    if SINGLE_SOCKET:
        handshake("sender")
        return [ttnn.MeshSocket(mesh, bigmesh_single_config())]
    sockets = []
    for i in range(NUM_DEVICES):
        handshake("sender")
        config = submesh_socket_config(i) if USE_SUBMESH else bigmesh_socket_config(i)
        sockets.append(ttnn.MeshSocket(mesh, config))
    return sockets


def sender():
    init()
    banner()
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, NUM_DEVICES), offset=ttnn.MeshCoordinate(0, 0))
    tensors = [make_sharded_tensor(mesh, j) for j in range(NUM_TENSORS)]

    # Send-side stats: confirm exactly what is on each device before the transfer.
    for j, tensor in enumerate(tensors):
        for i in range(NUM_DEVICES):
            mn, mx, me = tensor_stats(ttnn.to_torch(ttnn.get_device_tensors(tensor)[i]))
            log(f"[send] tensor {j} dev {i}: min={mn:g} max={mx:g} mean={me:g} expected={shard_value(i, j):g}")

    sockets = open_sender_sockets(mesh)
    log(f"{len(sockets)} socket(s) open")

    for j, tensor in enumerate(tensors):
        if SINGLE_SOCKET:
            ttnn.experimental.send_async(tensor, sockets[0])
        else:
            for socket in sockets:
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
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, NUM_DEVICES), offset=ttnn.MeshCoordinate(0, 0))

    submeshes = None
    if USE_SUBMESH:
        submeshes = parent.create_submeshes(ttnn.MeshShape(1, 1))
        sockets = []
        for i in range(NUM_DEVICES):
            handshake("receiver")
            sockets.append(ttnn.MeshSocket(submeshes[i], submesh_socket_config(i)))
        spec = submesh_recv_spec(submeshes[0])
    elif SINGLE_SOCKET:
        handshake("receiver")
        sockets = [ttnn.MeshSocket(parent, bigmesh_single_config())]
        spec = sharded_recv_spec(parent)
    else:
        sockets = []
        for i in range(NUM_DEVICES):
            handshake("receiver")
            sockets.append(ttnn.MeshSocket(parent, bigmesh_socket_config(i)))
        spec = sharded_recv_spec(parent)
    log(f"{len(sockets)} socket(s) open")

    # Issue every recv_async first, keep every template alive, then a SINGLE synchronize -- the
    # cadence that triggers the hang (this is deliberately NOT the per-key-sync workaround).
    if SINGLE_SOCKET:
        # recvd[j]: the one full-mesh template tensor j was received into (all N shards).
        recvd = [None] * NUM_TENSORS
        for j in range(NUM_TENSORS):
            template = ttnn.allocate_tensor_on_device(spec, parent)
            ttnn.experimental.recv_async(template, sockets[0])
            recvd[j] = template
            log_progress("recv issued", j)
    else:
        # recvd[i][j]: the template socket i received tensor j into.
        recvd = [[None] * NUM_TENSORS for _ in range(NUM_DEVICES)]
        for j in range(NUM_TENSORS):
            for i, socket in enumerate(sockets):
                device = submeshes[i] if USE_SUBMESH else parent
                template = ttnn.allocate_tensor_on_device(spec, device)
                ttnn.experimental.recv_async(template, socket)
                recvd[i][j] = template
            log_progress("recv issued", j)

    log("all recv_async issued; synchronizing ...")
    if USE_SUBMESH:
        for submesh in submeshes:
            ttnn.synchronize_device(submesh)
    else:
        ttnn.synchronize_device(parent)
    log("synced; verifying ...")

    def shard_of(i, j):
        """Device i's shard of tensor j, read back as a torch tensor for the current mode."""
        if SINGLE_SOCKET:
            return ttnn.to_torch(ttnn.get_device_tensors(recvd[j])[i])
        if USE_SUBMESH:
            return ttnn.to_torch(ttnn.get_device_tensors(recvd[i][j])[0])
        return ttnn.to_torch(ttnn.get_device_tensors(recvd[i][j])[i])

    all_correct = True
    for j in range(NUM_TENSORS):
        for i in range(NUM_DEVICES):
            got = shard_of(i, j)
            expected = torch.full(TENSOR_SHAPE, shard_value(i, j), dtype=torch.bfloat16)
            mn, mx, me = tensor_stats(got)
            correct, total = count_correct(got, expected)
            status = "OK" if correct == total else "CORRUPT"
            log(
                f"[recv] tensor {j} dev {i}: min={mn:g} max={mx:g} mean={me:g} "
                f"expected={shard_value(i, j):g} correct={correct}/{total} {status}"
            )
            if correct != total:
                all_correct = False

    log(f"all {NUM_DEVICES * NUM_TENSORS} tensors correct: {all_correct}")
    handshake("receiver")
    ttnn.close_mesh_device(parent)


if __name__ == "__main__":
    if int(os.environ["OMPI_COMM_WORLD_RANK"]) == SENDER_RANK:
        sender()
    else:
        receiver()

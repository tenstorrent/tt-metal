#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MeshSocket 4->4 repro on ONE mesh with 4 sockets -- reproduces a CORRUPT tensor.

Both ranks open a single [1, 4] mesh (no submeshes). The receiver opens NUM_DEVICES sockets
on that one mesh, socket i carrying a single connection sender(0, i) -> receiver(0, i). The
sender streams NUM_TENSORS sharded tensors, sending each tensor over all 4 sockets (socket i
transfers device i's shard). This is the "bigmesh_Nsock" pattern: 1 mesh, N sockets, each a
single (0,i)->(0,i) connection.

The receiver issues every recv_async first, then a single synchronize_device, then verifies.
This cadence reproduces the known corruption: some (device, tensor) shard comes back wrong
(historically the 3rd streamed tensor on devices 0/1), so the run reports MISMATCH and
"all ... tensors correct: False".

Each tensor's shard on device i is filled with shard_value(i, j) = i + 1 + j, distinct per
(device, tensor), so a misroute / reorder / stale-read is caught and localized.

Knobs:
  NUM_DEVICES   devices = sockets = 4 (must match the runner's [1, 4] mesh config).
  NUM_TENSORS   tensors streamed (env REPRO_NUM_TENSORS, default 4).
  TENSOR_SHAPE  per-device shard shape (env REPRO_TENSOR_SHAPE, default "1,1,32,32").
  REPRO_BRANCH  git branch label for logging (env, normally set by runner.sh).

Run: bash debug_mesh_socket/runner.sh   (from tt-train/sources/examples/grpo)
"""

import os
import subprocess

import torch
from ttnn._ttnn.multi_device import recv_bytes as _recv
from ttnn._ttnn.multi_device import send_bytes as _send

import ttnn

SENDER_RANK = 0
RECEIVER_RANK = 1
NUM_DEVICES = 4  # devices = sockets; both ranks open a [1, 4] mesh
NUM_TENSORS = int(os.environ.get("REPRO_NUM_TENSORS", "4"))
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


def shard_value(device_index, tensor_index):
    """Fill value for the shard on device `device_index` of tensor `tensor_index`.

    Distinct per (device, tensor) so a misroute or reordering is caught, and small enough that
    bf16 holds it exactly (keep NUM_DEVICES + NUM_TENSORS <= 256).
    """
    return float(device_index + 1 + tensor_index)


def handshake(role):
    """Two-rank rendezvous before each socket open: eager send, then blocking recv."""
    peer_rank, send_tag, recv_tag = (RECEIVER_RANK, 7, 8) if role == "sender" else (SENDER_RANK, 8, 7)
    _send(b"r", peer_rank, send_tag)
    _recv(1, peer_rank, recv_tag)


def socket_config(socket_index):
    """Socket `socket_index`: sender device (0, socket_index) -> receiver device (0, socket_index),
    on the SAME [1, 4] mesh on each side (no submeshes)."""
    core = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, socket_index), ttnn.CoreCoord(0, 0))
    memory = ttnn.SocketMemoryConfig(ttnn.BufferType.DRAM, SOCKET_FIFO_BYTES)
    return ttnn.SocketConfig(
        [ttnn.SocketConnection(core, core)],
        memory,
        sender_rank=SENDER_RANK,
        receiver_rank=RECEIVER_RANK,
    )


def open_socket(role, device, socket_index):
    handshake(role)
    socket = ttnn.MeshSocket(device, socket_config(socket_index))
    log(f"socket {socket_index} open")
    return socket


def init():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()


def open_mesh():
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, NUM_DEVICES), offset=ttnn.MeshCoordinate(0, 0))


def make_sharded_tensor(mesh, tensor_index):
    """One [NUM_DEVICES, *] tensor whose shard on device i is filled with shard_value(i, ...)."""
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
    """Spec of a [NUM_DEVICES, *] tensor sharded dim-0 across the [1, 4] mesh (one TENSOR_SHAPE
    shard per device), matching make_sharded_tensor -- used to allocate recv templates."""
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


def banner():
    log(f"START shape={TENSOR_SHAPE} num_tensors={NUM_TENSORS} num_devices={NUM_DEVICES} mode=bigmesh_Nsock")


def sender():
    init()
    banner()
    mesh = open_mesh()
    tensors = [make_sharded_tensor(mesh, j) for j in range(NUM_TENSORS)]
    sockets = [open_socket("sender", mesh, i) for i in range(NUM_DEVICES)]
    log(f"{len(sockets)} socket(s) open")

    for j, tensor in enumerate(tensors):
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
    parent = open_mesh()
    sockets = [open_socket("receiver", parent, i) for i in range(NUM_DEVICES)]
    log(f"{len(sockets)} socket(s) open")
    spec = sharded_recv_spec(parent)

    # recvd[i][j]: the [1, 4] template socket i received tensor j into (its device-i shard is
    # the meaningful one). Issue EVERY recv_async first, then a single synchronize -- the
    # cadence that reproduces the corruption.
    recvd = [[None] * NUM_TENSORS for _ in range(NUM_DEVICES)]
    for j in range(NUM_TENSORS):
        for i, socket in enumerate(sockets):
            template = ttnn.allocate_tensor_on_device(spec, parent)
            ttnn.experimental.recv_async(template, socket)
            recvd[i][j] = template
        log_progress("recv issued", j)

    log("all recv_async issued; synchronizing ...")
    ttnn.synchronize_device(parent)
    log("synced; verifying ...")

    all_correct = True
    for j in range(NUM_TENSORS):
        for i in range(NUM_DEVICES):
            got = ttnn.to_torch(ttnn.get_device_tensors(recvd[i][j])[i])
            expected = torch.full(TENSOR_SHAPE, shard_value(i, j), dtype=torch.bfloat16)
            correct = int(torch.eq(got, expected).sum().item())
            total = got.numel()
            status = "OK" if correct == total else "CORRUPT"
            mn, mx, me = got.float().min().item(), got.float().max().item(), got.float().mean().item()
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

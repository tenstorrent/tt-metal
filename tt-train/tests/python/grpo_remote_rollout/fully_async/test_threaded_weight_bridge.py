# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Two-rank tt-run test for :class:`ThreadedWeightBridge` (dict API).

Rank 0 (sender):
  - Opens a [1, 1] mesh with `num_command_queues=2`.
  - Allocates two live tensors ``x0`` (grows +1 per add) and ``x1`` (grows
    +2 per add).
  - Every ``PUSH_EVERY_S`` seconds, calls
    ``bridge.send_weights({"w0": x0, "w1": x1})``. Immediately after the
    call, does another burst of ``ttnn.add`` on the SAME ``x0`` / ``x1``.
    This proves the caller can mutate its live tensors right after
    ``send_weights`` returns: the ``ttnn.copy`` inside ``send_weights``
    froze the values into the on-device pads.
  - After ``N_PUSHES`` calls, ``bridge.close()`` emits the length-0
    manifest close message, then ships the expected first-elem summary
    for both keys to rank 1 on the world context (tags 998/999).

Rank 1 (receiver):
  - Opens a [1, 1] mesh with `num_command_queues=2`.
  - Loops on ``with bridge.receive_weights() as dicts:``. The recv pad
    lock is held across the ``with`` body, so the bridge cannot overwrite
    the recv pads while rank 1 reads them.
  - Samples the first elem of each key on CQ0, records both, then
    asserts the received trail matches rank 0's expected summary.

Run via ``runner_threaded_weight_bridge.sh``.
"""

from __future__ import annotations

import os

import pytest

_WORLD_SIZE = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "0"))
if _WORLD_SIZE != 2:
    pytest.skip(
        "test_threaded_weight_bridge must run under tt-run with world_size == 2 "
        "(use runner_threaded_weight_bridge.sh).",
        allow_module_level=True,
    )

_MPI_RANK = int(os.environ["OMPI_COMM_WORLD_RANK"])

import gc  # noqa: E402
import json  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402
from typing import List  # noqa: E402

import torch  # noqa: E402
import ttnn  # noqa: E402

from utils.threaded_weight_bridge import ThreadedWeightBridge  # noqa: E402


# ---- knobs -------------------------------------------------------------------
SENDER_RANK: int = 0
RECEIVER_RANK: int = 1

MESH_SHAPE: tuple = (1, 1)
NUM_CQS: int = 2
PAD_SHAPE: tuple = (64, 64)  # fp32 TILE-aligned, 16 KiB
PAD_TORCH_DTYPE = torch.float32
PAD_TTNN_DTYPE = ttnn.float32
N_PUSHES: int = 5
TICK_S: float = 1.0
PUSH_EVERY_S: float = 5.0
ADDS_PER_TICK: int = 1000

# Reserved MPI tags on the world context for the rank-0 -> rank-1 summary
# handoff. Disjoint from anything the bridge uses on its private context.
_TAG_SUMMARY_LEN: int = 998
_TAG_SUMMARY_BODY: int = 999


def _open_mesh() -> "ttnn.MeshDevice":
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*MESH_SHAPE),
        offset=ttnn.MeshCoordinate(0, 0),
        num_command_queues=NUM_CQS,
    )


def _fresh(mesh: "ttnn.MeshDevice", value: float) -> "ttnn.Tensor":
    return ttnn.from_torch(
        torch.full(PAD_SHAPE, value, dtype=PAD_TORCH_DTYPE),
        dtype=PAD_TTNN_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _rank0_side() -> None:
    print(f"[rank 0] opening [1, 1] mesh with num_command_queues={NUM_CQS}...", flush=True)
    mesh = _open_mesh()

    bridge = ThreadedWeightBridge.sender(peer_rank=RECEIVER_RANK, mesh_device=mesh)
    try:
        x0 = _fresh(mesh, 0.0)
        x1 = _fresh(mesh, 100.0)
        one = _fresh(mesh, 1.0)
        two = _fresh(mesh, 2.0)

        bridge.connect()
        print("[rank 0] bridge connected + sender thread started", flush=True)

        expected: List[dict] = []
        sent = 0
        last_push = time.monotonic()
        tick = 0
        while sent < N_PUSHES:
            t_burst_start = time.perf_counter()
            for _ in range(ADDS_PER_TICK):
                x0 = ttnn.add(x0, one)
                x1 = ttnn.add(x1, two)
            burst_ms = (time.perf_counter() - t_burst_start) * 1000.0

            now = time.monotonic()
            if now - last_push >= PUSH_EVERY_S:
                first0 = float(ttnn.to_torch(x0, cq_id=0)[0, 0])
                first1 = float(ttnn.to_torch(x1, cq_id=0)[0, 0])
                expected.append({"w0": first0, "w1": first1})

                # Push. `ttnn.copy` inside freezes the current values into
                # the pads; we are then free to mutate x0 / x1 immediately.
                bridge.send_weights({"w0": x0, "w1": x1})

                # Deliberate mutation of the SAME x0 / x1 right after the
                # push. If the bridge were reading x0 / x1 directly instead
                # of the pads, this burst would race with the bridge's
                # in-flight to_torch and corrupt the wire values.
                for _ in range(ADDS_PER_TICK):
                    x0 = ttnn.add(x0, one)
                    x1 = ttnn.add(x1, two)

                print(
                    f"[rank 0] pushed v={sent} w0[0,0]={first0} w1[0,0]={first1} "
                    f"(pre-push tick {tick} adds={ADDS_PER_TICK} took {burst_ms:.1f}ms; "
                    f"then another {ADDS_PER_TICK} mutating post-push burst)",
                    flush=True,
                )
                sent += 1
                last_push = now
            else:
                print(
                    f"[rank 0] tick {tick}: {ADDS_PER_TICK} adds took {burst_ms:.1f}ms",
                    flush=True,
                )
            tick += 1
            time.sleep(TICK_S)

        print("[rank 0] closing bridge (close message + join)...", flush=True)
        bridge.close()
        print("[rank 0] bridge closed", flush=True)

        body = json.dumps(expected).encode()
        ttnn.distributed_context_send_bytes(struct.pack("<I", len(body)), RECEIVER_RANK, _TAG_SUMMARY_LEN)
        ttnn.distributed_context_send_bytes(body, RECEIVER_RANK, _TAG_SUMMARY_BODY)
        print(f"[rank 0] shipped expected summary ({len(expected)} entries)", flush=True)
    finally:
        gc.collect()
        try:
            ttnn.close_mesh_device(mesh)
        except Exception as e:  # noqa: BLE001
            print(f"[rank 0] close_mesh_device: {type(e).__name__}: {e}", flush=True)


def _rank1_side() -> None:
    print(f"[rank 1] opening [1, 1] mesh with num_command_queues={NUM_CQS}...", flush=True)
    mesh = _open_mesh()

    bridge = ThreadedWeightBridge.receiver(peer_rank=SENDER_RANK, mesh_device=mesh)
    try:
        bridge.connect()
        print("[rank 1] bridge connected + receiver thread started", flush=True)

        received: List[dict] = []
        while True:
            # Context manager holds the recv pad lock across the with body.
            # Bridge is blocked from overwriting the pads until we exit.
            with bridge.receive_weights() as dicts:
                dev_dict = dicts[0]
                if not dev_dict:
                    break  # peer closed and no pending
                entry = {}
                for k in sorted(dev_dict.keys()):
                    entry[k] = float(ttnn.to_torch(dev_dict[k], cq_id=0)[0, 0])
            received.append(entry)
            print(
                f"[rank 1] received v={len(received) - 1} " + " ".join(f"{k}[0,0]={v}" for k, v in entry.items()),
                flush=True,
            )

        print("[rank 1] recv loop drained; reading expected summary from rank 0...", flush=True)
        (n,) = struct.unpack("<I", ttnn.distributed_context_recv_bytes(4, SENDER_RANK, _TAG_SUMMARY_LEN))
        expected = json.loads(ttnn.distributed_context_recv_bytes(int(n), SENDER_RANK, _TAG_SUMMARY_BODY).decode())

        bridge.close()

        assert expected == received, f"expected={expected} got={received}"
    finally:
        gc.collect()
        try:
            ttnn.close_mesh_device(mesh)
        except Exception as e:  # noqa: BLE001
            print(f"[rank 1] close_mesh_device: {type(e).__name__}: {e}", flush=True)


def test_threaded_weight_bridge() -> None:
    """End-to-end dict-API round-trip on the ThreadedWeightBridge."""
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()
    if _MPI_RANK == SENDER_RANK:
        _rank0_side()
    else:
        _rank1_side()

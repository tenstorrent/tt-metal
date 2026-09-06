# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Two-rank tt-run test for :class:`ThreadedWeightBridge`.

Rank 0 (sender):
  - Opens a [1, 1] mesh with `num_command_queues=2`.
  - Allocates a live tensor ``x`` (bf16 TILE DRAM on CQ0).
  - Loops on a 1-second tick, running ``x = ttnn.add(x, one)`` on CQ0 every
    tick. Every 5 seconds (or every 5th tick, whichever the constants set)
    calls ``bridge.push_tensor(x)`` which D->D-copies ``x`` into the send
    pad on CQ0 and hands off to the sender bridge thread.
  - After ``N_PUSHES`` messages, calls ``bridge.close()`` which sends the
    length-0 sentinel; then MPI-sends the "expected values" summary to
    rank 1 on the world context so rank 1 can print PASS/FAIL.

Rank 1 (receiver):
  - Opens a [1, 1] mesh with `num_command_queues=2`.
  - Loops calling ``pad = bridge.acquire_recv_pad()``. Under the pad lock,
    the bridge has already ``wait_for_event(0, ...)``'d, so the pad is
    coherent on CQ0. Samples ``float(ttnn.to_torch(pad, cq_id=0)[0, 0])``,
    then calls ``bridge.release_recv_pad()``.
  - Breaks on ``pad is None`` (peer closed).
  - Reads the expected-values summary from rank 0 and prints ``[PASS]`` or
    ``[FAIL]``.
"""

from __future__ import annotations

import gc
import io
import json
import struct
import sys
import time
from pathlib import Path
from typing import List

import torch

# Make this file's own directory importable so `weight_bridge` resolves.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import ttnn  # noqa: E402

from weight_bridge import ThreadedWeightBridge  # noqa: E402


# ---- knobs -------------------------------------------------------------------
SENDER_RANK: int = 0
RECEIVER_RANK: int = 1

MESH_SHAPE: tuple = (1, 1)
NUM_CQS: int = 2
PAD_SHAPE: tuple = (64, 64)  # fp32 TILE-aligned, 16 KiB
# fp32 (not bf16) so that ADDS_PER_TICK * N_PUSHES accumulations don't
# saturate to a constant value. bf16 has ~8 mantissa bits; once `x` reaches
# ~256, `x + 1` rounds off. fp32 has 23 mantissa bits and stays exact into
# the millions.
PAD_TORCH_DTYPE = torch.float32
PAD_TTNN_DTYPE = ttnn.float32
N_PUSHES: int = 5
TICK_S: float = 1.0
PUSH_EVERY_S: float = 5.0

# How many ttnn.add calls the rank-0 main thread issues per tick. Keeps CQ0
# busy while the bridge thread runs D->H on CQ1, so we exercise the multi-CQ
# overlap path and not just a single-op-per-tick smoke test.
ADDS_PER_TICK: int = 1000

# Reserved MPI tags on the world context for the rank-0 -> rank-1 handoff of
# the "expected values" JSON summary. Disjoint from anything the bridge uses
# on the duplicated context.
_TAG_SUMMARY_LEN: int = 998
_TAG_SUMMARY_BODY: int = 999


def _open_mesh() -> "ttnn.MeshDevice":
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*MESH_SHAPE),
        offset=ttnn.MeshCoordinate(0, 0),
        num_command_queues=NUM_CQS,
    )


def _rank0_main() -> None:
    print(f"[rank 0] opening [1, 1] mesh with num_command_queues={NUM_CQS}...", flush=True)
    mesh = _open_mesh()

    bridge = ThreadedWeightBridge.sender(
        peer_rank=RECEIVER_RANK,
        mesh_device=mesh,
        shape=PAD_SHAPE,
        dtype=PAD_TTNN_DTYPE,
    )
    try:
        # Live tensor we mutate with ttnn.add and push periodically.
        x = ttnn.from_torch(
            torch.zeros(*PAD_SHAPE, dtype=PAD_TORCH_DTYPE),
            dtype=PAD_TTNN_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        one = ttnn.from_torch(
            torch.ones(*PAD_SHAPE, dtype=PAD_TORCH_DTYPE),
            dtype=PAD_TTNN_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        bridge.connect()
        print("[rank 0] bridge connected + sender thread started", flush=True)

        expected: List[float] = []
        sent = 0
        last_push = time.monotonic()
        tick = 0
        while sent < N_PUSHES:
            # Keep CQ0 busy with a burst of adds. `x` accumulates one per add,
            # so its value drifts by ADDS_PER_TICK per tick. Each ttnn.add
            # returns a NEW tensor, so the previous `x` (if it's still sitting
            # in the send pad from a prior push) is untouched.
            t_burst_start = time.perf_counter()
            for _ in range(ADDS_PER_TICK):
                x = ttnn.add(x, one)
            burst_ms = (time.perf_counter() - t_burst_start) * 1000.0

            now = time.monotonic()
            if now - last_push >= PUSH_EVERY_S:
                # Snapshot the "expected first element" BEFORE pushing so we
                # can verify the round-trip end-to-end. Cheap: CQ0 D->H of
                # the whole tensor and read one element.
                first_elem = float(ttnn.to_torch(x, cq_id=0)[0, 0])
                expected.append(first_elem)

                bridge.push_tensor(x)  # D->D under send_pad_lock, CQ0
                print(
                    f"[rank 0] pushed v={sent} first_elem={first_elem} "
                    f"(after tick {tick}, {ADDS_PER_TICK} adds took {burst_ms:.1f}ms)",
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

        print("[rank 0] closing bridge (drains + sentinel + join)...", flush=True)
        bridge.close()
        print("[rank 0] bridge closed", flush=True)

        # Ship the "expected values" summary to rank 1 on the world context.
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


def _rank1_main() -> None:
    print(f"[rank 1] opening [1, 1] mesh with num_command_queues={NUM_CQS}...", flush=True)
    mesh = _open_mesh()

    bridge = ThreadedWeightBridge.receiver(
        peer_rank=SENDER_RANK,
        mesh_device=mesh,
        shape=PAD_SHAPE,
        dtype=PAD_TTNN_DTYPE,
    )
    try:
        bridge.connect()
        print("[rank 1] bridge connected + receiver thread started", flush=True)

        received: List[float] = []
        while True:
            pad = bridge.acquire_recv_pad()  # holds recv_pad_lock on return
            if pad is None:
                break
            try:
                # CQ0 sample under the lock; bridge has already wait_for_event'd
                # so this is coherent with the CQ1 write it just did.
                first_elem = float(ttnn.to_torch(pad, cq_id=0)[0, 0])
            finally:
                bridge.release_recv_pad()

            received.append(first_elem)
            print(
                f"[rank 1] received v={len(received)-1} first_elem={first_elem}",
                flush=True,
            )

        print("[rank 1] recv loop drained; reading expected summary from rank 0...", flush=True)
        (n,) = struct.unpack(
            "<I",
            ttnn.distributed_context_recv_bytes(4, SENDER_RANK, _TAG_SUMMARY_LEN),
        )
        expected = json.loads(ttnn.distributed_context_recv_bytes(int(n), SENDER_RANK, _TAG_SUMMARY_BODY).decode())

        ok = expected == received
        tag = "[PASS]" if ok else "[FAIL]"
        print(f"[rank 1] {tag} expected={expected} got={received}", flush=True)

        # Also close the bridge so the receiver-side state is tidied. The
        # receiver bridge thread has already exited (via the sentinel path)
        # so close() is basically a no-op notify.
        bridge.close()
    finally:
        gc.collect()
        try:
            ttnn.close_mesh_device(mesh)
        except Exception as e:  # noqa: BLE001
            print(f"[rank 1] close_mesh_device: {type(e).__name__}: {e}", flush=True)


def main() -> None:
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    world_size = int(ttnn.distributed_context_get_size())
    if world_size != 2:
        raise RuntimeError(
            f"test_threaded_bridge must run under tt-run with world_size == 2 (got {world_size}). " "Use runner.sh."
        )

    rank = int(ttnn.distributed_context_get_rank())
    if rank == SENDER_RANK:
        _rank0_main()
    elif rank == RECEIVER_RANK:
        _rank1_main()
    else:
        raise RuntimeError(
            f"Unexpected MPI rank {rank} (world_size={world_size}); "
            f"expected exactly two ranks: SENDER={SENDER_RANK}, RECEIVER={RECEIVER_RANK}."
        )


if __name__ == "__main__":
    main()

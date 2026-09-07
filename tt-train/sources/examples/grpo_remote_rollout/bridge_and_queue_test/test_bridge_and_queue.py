# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Two-rank tt-run test that runs the ThreadedWeightBridge and the
RolloutQueue concurrently on separate duplicated MPI contexts.

Rank directions match the real fully-async trainer:

  * Rank 0 = TTML trainer.
      - ``ThreadedWeightBridge.sender``: ships fresh weights to rank 1.
      - ``RolloutQueue.consumer``: reads incoming rollout batches.
  * Rank 1 = TTT rollout worker.
      - ``ThreadedWeightBridge.receiver``: receives weights.
      - ``RolloutQueue.producer``: pushes freshly generated batches.

Rank 0 main loop (mirrors the training loop):

  1. Pop a rollout batch from RolloutQueue (need data before training).
  2. Run a burst of ttnn.add on CQ0 (mock gradient work).
  3. Every other round, push the updated weight tensor through the bridge.

Rank 1 main loop:

  1. Push a fake rollout batch through RolloutQueue.
  2. Between rounds, non-blockingly check the recv pad for a fresh
     weight update; if one arrived, sample + verify + release.

Verification: rank 1 ships two expected-checksum lists to rank 0 at the
end (one per channel). Rank 0 compares and prints one PASS/FAIL per
channel plus a top-level [COMBINED PASS] / [COMBINED FAIL].

Purpose: proves the two duplicated MPI contexts truly progress in
parallel under MPI_THREAD_MULTIPLE. If the ttml DistributedContext
bindings ever regress and stop releasing the GIL, this test will
deadlock in a way that ``test_threaded_bridge.py`` and
``test_rollout_queue.py`` alone might not catch.
"""

from __future__ import annotations

import gc
import json
import struct
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import ttnn  # noqa: E402

from rollout_queue import RolloutBatch, RolloutQueue  # noqa: E402
from weight_bridge import ThreadedWeightBridge  # noqa: E402


# ---- knobs -------------------------------------------------------------------
TTML_RANK: int = 0  # weight sender + rollout consumer
TTT_RANK: int = 1  # weight receiver + rollout producer

MESH_SHAPE: tuple = (1, 1)
NUM_CQS: int = 2

# Weight bridge pad shape (matches test_threaded_bridge.py).
PAD_SHAPE: tuple = (64, 64)
PAD_TTNN_DTYPE = ttnn.float32
PAD_TORCH_DTYPE = torch.float32

# Rollout batch shape.
BATCH_B: int = 4
MAX_COMPLETION_LEN: int = 32
PROMPT_LEN: int = 6

N_ROUNDS: int = 6
QUEUE_CAPACITY: int = 2

# Ticks (ttnn.add calls) per round on rank 0 -- mock gradient work.
ADDS_PER_ROUND: int = 200

# Reserved MPI tags on the world context for the end-of-test summary handoff.
_TAG_QUEUE_SUMMARY_LEN: int = 998
_TAG_QUEUE_SUMMARY_BODY: int = 999
_TAG_BRIDGE_SUMMARY_LEN: int = 996
_TAG_BRIDGE_SUMMARY_BODY: int = 997


def _open_mesh() -> "ttnn.MeshDevice":
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*MESH_SHAPE),
        offset=ttnn.MeshCoordinate(0, 0),
        num_command_queues=NUM_CQS,
    )


def _build_batch(batch_id: int, weight_version: int) -> RolloutBatch:
    """Deterministic fake batch keyed on batch_id."""
    torch.manual_seed(1000 + batch_id)
    prompts = [[(batch_id * 100 + p * 10 + t) % 32000 for t in range(PROMPT_LEN)] for p in range(BATCH_B)]
    completions = [
        [(batch_id * 200 + p * 20 + t) % 32000 for t in range((batch_id + p + 1) % MAX_COMPLETION_LEN + 1)]
        for p in range(BATCH_B)
    ]
    logprobs = torch.randn(BATCH_B, MAX_COMPLETION_LEN, dtype=torch.float32) * 0.1
    return RolloutBatch(
        batch_id=batch_id,
        weight_version=weight_version,
        prompts=prompts,
        completions=completions,
        logprobs=logprobs,
    )


def _batch_checksum(batch: RolloutBatch) -> float:
    tok_hash = sum(len(p) + len(c) * 10 for p, c in zip(batch.prompts, batch.completions))
    return float(batch.logprobs.sum().item()) + float(tok_hash)


def _ttml_main() -> None:
    """Rank 0: weight bridge sender + rollout queue consumer."""
    print(f"[rank {TTML_RANK}] TTML: opening mesh with num_command_queues={NUM_CQS}...", flush=True)
    mesh = _open_mesh()

    bridge = ThreadedWeightBridge.sender(
        peer_rank=TTT_RANK,
        mesh_device=mesh,
        shape=PAD_SHAPE,
        dtype=PAD_TTNN_DTYPE,
    )
    q = RolloutQueue.consumer(peer_rank=TTT_RANK, capacity=QUEUE_CAPACITY)

    try:
        # Live "weights" tensor for the bridge sender path.
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

        # Order matters: bridge.connect() and q.connect() are collective on
        # duplicated MPI contexts; both ranks must call in the same order.
        bridge.connect()
        q.connect()
        print(f"[rank {TTML_RANK}] TTML: both channels connected", flush=True)

        received_rollout_checksums: List[float] = []
        pushed_weight_first_elems: List[float] = []

        for r in range(N_ROUNDS):
            # 1) Pop a rollout batch (need data before training).
            batch = q.pop()
            if batch is None:
                print(f"[rank {TTML_RANK}] TTML: queue closed early at round {r}; breaking", flush=True)
                break
            received_rollout_checksums.append(_batch_checksum(batch))
            print(
                f"[rank {TTML_RANK}] round {r}: popped rollout batch {batch.batch_id} "
                f"wver={batch.weight_version} checksum={received_rollout_checksums[-1]:.4f}",
                flush=True,
            )

            # 2) Mock gradient work on CQ0.
            t0 = time.perf_counter()
            for _ in range(ADDS_PER_ROUND):
                x = ttnn.add(x, one)
            burst_ms = (time.perf_counter() - t0) * 1000.0

            # 3) Every other round, push the updated weight tensor.
            if r % 2 == 0:
                first_elem = float(ttnn.to_torch(x, cq_id=0)[0, 0])
                pushed_weight_first_elems.append(first_elem)
                bridge.push_tensor(x)
                print(
                    f"[rank {TTML_RANK}] round {r}: pushed weight ({ADDS_PER_ROUND} adds "
                    f"took {burst_ms:.1f}ms, first_elem={first_elem})",
                    flush=True,
                )
            else:
                print(
                    f"[rank {TTML_RANK}] round {r}: {ADDS_PER_ROUND} adds took {burst_ms:.1f}ms "
                    f"(no push this round)",
                    flush=True,
                )

        print(f"[rank {TTML_RANK}] TTML: main loop done; closing bridge", flush=True)
        bridge.close()

        # Read verification summaries shipped by rank 1.
        (n,) = struct.unpack("<I", ttnn.distributed_context_recv_bytes(4, TTT_RANK, _TAG_QUEUE_SUMMARY_LEN))
        expected_rollout = json.loads(
            ttnn.distributed_context_recv_bytes(int(n), TTT_RANK, _TAG_QUEUE_SUMMARY_BODY).decode()
        )
        (n,) = struct.unpack("<I", ttnn.distributed_context_recv_bytes(4, TTT_RANK, _TAG_BRIDGE_SUMMARY_LEN))
        expected_bridge = json.loads(
            ttnn.distributed_context_recv_bytes(int(n), TTT_RANK, _TAG_BRIDGE_SUMMARY_BODY).decode()
        )

        queue_ok = expected_rollout == received_rollout_checksums
        # The bridge summary is the list of first_elem values we pushed;
        # rank 1 confirms which ones it observed. We just check that our
        # pushed list matches rank 1's observed list.
        bridge_ok = expected_bridge == pushed_weight_first_elems

        print(
            f"[rank {TTML_RANK}] rollout channel: {'PASS' if queue_ok else 'FAIL'} "
            f"expected={[round(x, 4) for x in expected_rollout]} "
            f"got={[round(x, 4) for x in received_rollout_checksums]}",
            flush=True,
        )
        print(
            f"[rank {TTML_RANK}] bridge channel:  {'PASS' if bridge_ok else 'FAIL'} "
            f"pushed={pushed_weight_first_elems} rank1_saw={expected_bridge}",
            flush=True,
        )

        overall_ok = queue_ok and bridge_ok
        print(
            f"[rank {TTML_RANK}] {'[COMBINED PASS]' if overall_ok else '[COMBINED FAIL]'}",
            flush=True,
        )

        q.close()
    finally:
        gc.collect()
        try:
            ttnn.close_mesh_device(mesh)
        except Exception as e:  # noqa: BLE001
            print(f"[rank {TTML_RANK}] close_mesh_device: {type(e).__name__}: {e}", flush=True)


def _ttt_main() -> None:
    """Rank 1: weight bridge receiver + rollout queue producer."""
    print(f"[rank {TTT_RANK}] TTT: opening mesh with num_command_queues={NUM_CQS}...", flush=True)
    mesh = _open_mesh()

    bridge = ThreadedWeightBridge.receiver(
        peer_rank=TTML_RANK,
        mesh_device=mesh,
        shape=PAD_SHAPE,
        dtype=PAD_TTNN_DTYPE,
    )
    q = RolloutQueue.producer(peer_rank=TTML_RANK, capacity=QUEUE_CAPACITY)

    try:
        bridge.connect()
        q.connect()
        print(f"[rank {TTT_RANK}] TTT: both channels connected", flush=True)

        pushed_rollout_checksums: List[float] = []
        observed_weight_first_elems: List[float] = []

        for r in range(N_ROUNDS):
            # 1) Push a fake rollout batch.
            batch = _build_batch(batch_id=r, weight_version=r // 2)
            pushed_rollout_checksums.append(_batch_checksum(batch))
            q.push(batch)
            print(
                f"[rank {TTT_RANK}] round {r}: pushed rollout batch {batch.batch_id} "
                f"wver={batch.weight_version} checksum={pushed_rollout_checksums[-1]:.4f}",
                flush=True,
            )

            # 2) Between rounds, check for a fresh weight update from rank 0.
            #    Try a bounded number of times so we don't block forever if
            #    rank 0 is running an odd-round (no push).
            _drain_weight_pad_if_ready(bridge, observed_weight_first_elems, max_attempts=5)

            time.sleep(0.05)

        # After the main loop, drain any remaining pending weight updates.
        # Rank 0 might have pushed a weight in its last "even" round after
        # we already advanced past it.
        _drain_weight_pad_if_ready(bridge, observed_weight_first_elems, max_attempts=20)

        print(f"[rank {TTT_RANK}] TTT: main loop done; closing queue", flush=True)
        q.close()

        # Ship two summaries so rank 0 can print PASS/FAIL for both channels.
        body = json.dumps(pushed_rollout_checksums).encode()
        ttnn.distributed_context_send_bytes(struct.pack("<I", len(body)), TTML_RANK, _TAG_QUEUE_SUMMARY_LEN)
        ttnn.distributed_context_send_bytes(body, TTML_RANK, _TAG_QUEUE_SUMMARY_BODY)

        body = json.dumps(observed_weight_first_elems).encode()
        ttnn.distributed_context_send_bytes(struct.pack("<I", len(body)), TTML_RANK, _TAG_BRIDGE_SUMMARY_LEN)
        ttnn.distributed_context_send_bytes(body, TTML_RANK, _TAG_BRIDGE_SUMMARY_BODY)
        print(f"[rank {TTT_RANK}] TTT: shipped both summaries", flush=True)

        bridge.close()
    finally:
        gc.collect()
        try:
            ttnn.close_mesh_device(mesh)
        except Exception as e:  # noqa: BLE001
            print(f"[rank {TTT_RANK}] close_mesh_device: {type(e).__name__}: {e}", flush=True)


def _drain_weight_pad_if_ready(
    bridge: "ThreadedWeightBridge",
    observed: List[float],
    *,
    max_attempts: int,
) -> None:
    """Consume as many pending weight pads as are currently ready.

    The bridge's ``acquire_recv_pad`` blocks until a message arrives, so we
    use ``has_data`` peek + a short spin. If nothing is pending after
    ``max_attempts`` tiny sleeps, we give up and return; a future round
    will pick up whatever arrived after.
    """
    for _ in range(max_attempts):
        # Grab the pad only if the receiver bridge has landed something.
        # There is no non-blocking primitive on the bridge yet; approximate
        # it by peeking the has_data flag (unsafe but harmless in this test)
        # and only calling acquire when we expect it to return promptly.
        if not getattr(bridge, "_has_data", False):
            time.sleep(0.01)
            continue
        pad = bridge.acquire_recv_pad()
        if pad is None:
            return
        try:
            first_elem = float(ttnn.to_torch(pad, cq_id=0)[0, 0])
            observed.append(first_elem)
            print(
                f"[rank {TTT_RANK}] observed fresh weight first_elem={first_elem}",
                flush=True,
            )
        finally:
            bridge.release_recv_pad()


def main() -> None:
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    world_size = int(ttnn.distributed_context_get_size())
    if world_size != 2:
        raise RuntimeError(
            f"test_bridge_and_queue must run under tt-run with world_size == 2 (got {world_size}). "
            "Use runner_bridge_and_queue.sh."
        )

    rank = int(ttnn.distributed_context_get_rank())
    if rank == TTML_RANK:
        _ttml_main()
    elif rank == TTT_RANK:
        _ttt_main()
    else:
        raise RuntimeError(f"Unexpected MPI rank {rank}; expected 0 or 1.")


if __name__ == "__main__":
    main()

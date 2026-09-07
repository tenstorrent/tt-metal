# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Two-rank tt-run test for :class:`RolloutQueue`.

Rank 1 (producer):
  - Builds ``N_BATCHES`` deterministic RolloutBatch objects.
  - Pushes each one through the queue.
  - Records a per-batch checksum before pushing and ships the whole list
    of checksums to rank 0 at the end (world context, tags 998 / 999).

Rank 0 (consumer):
  - Pops batches until pop() returns None.
  - Computes the checksum for each received batch.
  - Reads the expected checksums from rank 1.
  - Prints [PASS] / [FAIL].

Also visible in the log: per-push wall-clock. If ``CONSUMER_SLEEP_S > 0``
and ``CAPACITY`` is small, the producer's push time should grow once the
pipeline fills up, confirming Policy A back-pressure.
"""

from __future__ import annotations

import gc
import json
import struct
import sys
import time
from pathlib import Path
from typing import List

import torch

# Make this file's own directory importable so `rollout_queue` resolves.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import ttnn  # noqa: E402

from rollout_queue import RolloutBatch, RolloutQueue  # noqa: E402


# ---- knobs -------------------------------------------------------------------
PRODUCER_RANK: int = 1  # TTT rollout worker in the real system
CONSUMER_RANK: int = 0  # TTML trainer in the real system

MESH_SHAPE: tuple = (1, 1)
NUM_CQS: int = 1  # queue is host-only, one CQ is enough for
# tt-run's fabric handshake

N_BATCHES: int = 10
BATCH_B: int = 4  # completions per batch
MAX_COMPLETION_LEN: int = 32  # per-completion max token count
PROMPT_LEN: int = 6  # constant per batch (deterministic)

CAPACITY: int = 2  # local-queue max size on each rank
CONSUMER_SLEEP_S: float = 0.25  # slow consumer -> visible back-pressure
PRODUCER_GAP_S: float = 0.0  # producer pushes tight

# Reserved MPI tags on the world context for the producer -> consumer handoff
# of the "expected checksums" JSON summary. Disjoint from anything the queue
# uses on the duplicated context.
_TAG_SUMMARY_LEN: int = 998
_TAG_SUMMARY_BODY: int = 999


def _open_mesh() -> "ttnn.MeshDevice":
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*MESH_SHAPE),
        offset=ttnn.MeshCoordinate(0, 0),
        num_command_queues=NUM_CQS,
    )


def _build_batch(batch_id: int, weight_version: int) -> RolloutBatch:
    """Deterministic fake batch. Same batch_id -> same tokens and logprobs
    on both ranks, so both sides can compute the same checksum."""
    torch.manual_seed(1000 + batch_id)
    prompts = [[(batch_id * 100 + p * 10 + t) % 32000 for t in range(PROMPT_LEN)] for p in range(BATCH_B)]
    completions = [
        # Ragged completions of different lengths: batch_id + p + 1 tokens.
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
    """Cheap and stable checksum: sum of logprobs + hash of token counts."""
    tok_hash = sum(len(p) + len(c) * 10 for p, c in zip(batch.prompts, batch.completions))
    return float(batch.logprobs.sum().item()) + float(tok_hash)


def _rank_producer_main() -> None:
    print(f"[rank {PRODUCER_RANK}] producer: opening [1, 1] mesh with num_command_queues={NUM_CQS}...", flush=True)
    mesh = _open_mesh()
    q = RolloutQueue.producer(peer_rank=CONSUMER_RANK, capacity=CAPACITY)
    try:
        q.connect()
        print(f"[rank {PRODUCER_RANK}] producer connected + transport thread started", flush=True)

        expected_checksums: List[float] = []
        for i in range(N_BATCHES):
            batch = _build_batch(batch_id=i, weight_version=i // 3)
            expected_checksums.append(_batch_checksum(batch))

            t0 = time.perf_counter()
            q.push(batch)
            push_ms = (time.perf_counter() - t0) * 1000.0
            print(
                f"[rank {PRODUCER_RANK}] pushed batch {i} weight_version={i // 3} "
                f"checksum={expected_checksums[-1]:.4f} push_wall={push_ms:.1f}ms qsize={q.qsize()}",
                flush=True,
            )
            if PRODUCER_GAP_S > 0:
                time.sleep(PRODUCER_GAP_S)

        print(f"[rank {PRODUCER_RANK}] all batches pushed; closing queue", flush=True)
        q.close()
        print(f"[rank {PRODUCER_RANK}] queue closed", flush=True)

        # Ship the expected checksums for verification.
        body = json.dumps(expected_checksums).encode()
        ttnn.distributed_context_send_bytes(struct.pack("<I", len(body)), CONSUMER_RANK, _TAG_SUMMARY_LEN)
        ttnn.distributed_context_send_bytes(body, CONSUMER_RANK, _TAG_SUMMARY_BODY)
        print(f"[rank {PRODUCER_RANK}] shipped expected summary ({len(expected_checksums)} entries)", flush=True)
    finally:
        gc.collect()
        try:
            ttnn.close_mesh_device(mesh)
        except Exception as e:  # noqa: BLE001
            print(f"[rank {PRODUCER_RANK}] close_mesh_device: {type(e).__name__}: {e}", flush=True)


def _rank_consumer_main() -> None:
    print(f"[rank {CONSUMER_RANK}] consumer: opening [1, 1] mesh with num_command_queues={NUM_CQS}...", flush=True)
    mesh = _open_mesh()
    q = RolloutQueue.consumer(peer_rank=PRODUCER_RANK, capacity=CAPACITY)
    try:
        q.connect()
        print(f"[rank {CONSUMER_RANK}] consumer connected + transport thread started", flush=True)

        received_checksums: List[float] = []
        while True:
            batch = q.pop()
            if batch is None:
                break
            received_checksums.append(_batch_checksum(batch))
            print(
                f"[rank {CONSUMER_RANK}] popped batch {batch.batch_id} "
                f"weight_version={batch.weight_version} "
                f"B={len(batch.completions)} logprobs_shape={tuple(batch.logprobs.shape)} "
                f"checksum={received_checksums[-1]:.4f} qsize={q.qsize()}",
                flush=True,
            )
            if CONSUMER_SLEEP_S > 0:
                time.sleep(CONSUMER_SLEEP_S)

        print(
            f"[rank {CONSUMER_RANK}] pop loop drained; reading expected summary from rank {PRODUCER_RANK}...",
            flush=True,
        )

        (n,) = struct.unpack(
            "<I",
            ttnn.distributed_context_recv_bytes(4, PRODUCER_RANK, _TAG_SUMMARY_LEN),
        )
        expected = json.loads(ttnn.distributed_context_recv_bytes(int(n), PRODUCER_RANK, _TAG_SUMMARY_BODY).decode())

        ok = len(expected) == len(received_checksums) and all(
            abs(a - b) < 1e-3 for a, b in zip(expected, received_checksums)
        )
        tag = "[PASS]" if ok else "[FAIL]"
        print(
            f"[rank {CONSUMER_RANK}] {tag} "
            f"expected={[round(x, 4) for x in expected]} "
            f"got={[round(x, 4) for x in received_checksums]}",
            flush=True,
        )

        q.close()
    finally:
        gc.collect()
        try:
            ttnn.close_mesh_device(mesh)
        except Exception as e:  # noqa: BLE001
            print(f"[rank {CONSUMER_RANK}] close_mesh_device: {type(e).__name__}: {e}", flush=True)


def main() -> None:
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    world_size = int(ttnn.distributed_context_get_size())
    if world_size != 2:
        raise RuntimeError(
            f"test_rollout_queue must run under tt-run with world_size == 2 (got {world_size}). "
            "Use runner_rollout_queue.sh."
        )

    rank = int(ttnn.distributed_context_get_rank())
    if rank == PRODUCER_RANK:
        _rank_producer_main()
    elif rank == CONSUMER_RANK:
        _rank_consumer_main()
    else:
        raise RuntimeError(f"Unexpected MPI rank {rank}; expected {PRODUCER_RANK} or {CONSUMER_RANK}.")


if __name__ == "__main__":
    main()

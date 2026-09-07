# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Two-rank tt-run test for :class:`RolloutQueue`.

Rank 1 (producer):
  - Builds ``N_BATCHES`` deterministic ``RolloutBatch`` objects.
  - Pushes each one through the queue.
  - Records a per-batch checksum before pushing and ships the whole list
    to rank 0 at the end (world context, tags 998 / 999).

Rank 0 (consumer):
  - Pops batches until ``pop()`` returns ``None``.
  - Computes the checksum for each received batch.
  - Reads the expected checksums from rank 1 and asserts they match.

Also visible in the log at ``pytest -s``: per-push wall-clock. If
``CONSUMER_SLEEP_S > 0`` and ``CAPACITY`` is small, the producer's push
time grows once the pipeline fills up (Policy A back-pressure).

Run via ``runner_rollout_queue.sh``.
"""

from __future__ import annotations

import os

import pytest

_WORLD_SIZE = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "0"))
if _WORLD_SIZE != 2:
    pytest.skip(
        "test_rollout_queue must run under tt-run with world_size == 2 (use runner_rollout_queue.sh).",
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

from utils.rollout_queue import RolloutBatch, RolloutQueue  # noqa: E402


# ---- knobs -------------------------------------------------------------------
PRODUCER_RANK: int = 1  # TTT rollout worker in the real system
CONSUMER_RANK: int = 0  # TTML trainer in the real system

MESH_SHAPE: tuple = (1, 1)
NUM_CQS: int = 1  # queue is host-only; one CQ is enough for the fabric handshake

N_BATCHES: int = 10
BATCH_B: int = 4
MAX_COMPLETION_LEN: int = 32
PROMPT_LEN: int = 6

CAPACITY: int = 2
CONSUMER_SLEEP_S: float = 0.25
PRODUCER_GAP_S: float = 0.0

_TAG_SUMMARY_LEN: int = 998
_TAG_SUMMARY_BODY: int = 999


def _open_mesh() -> "ttnn.MeshDevice":
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*MESH_SHAPE),
        offset=ttnn.MeshCoordinate(0, 0),
        num_command_queues=NUM_CQS,
    )


def _build_batch(batch_id: int, weight_version: int) -> RolloutBatch:
    """Deterministic fake batch. Same batch_id -> same tokens and
    logprobs on both ranks, so both sides can compute the same checksum."""
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
    """Cheap and stable checksum: sum of logprobs + hash of token counts."""
    tok_hash = sum(len(p) + len(c) * 10 for p, c in zip(batch.prompts, batch.completions))
    return float(batch.logprobs.sum().item()) + float(tok_hash)


def _rank_producer_side() -> None:
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


def _rank_consumer_side() -> None:
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

        (n,) = struct.unpack("<I", ttnn.distributed_context_recv_bytes(4, PRODUCER_RANK, _TAG_SUMMARY_LEN))
        expected = json.loads(ttnn.distributed_context_recv_bytes(int(n), PRODUCER_RANK, _TAG_SUMMARY_BODY).decode())

        q.close()

        assert len(expected) == len(received_checksums) and all(
            abs(a - b) < 1e-3 for a, b in zip(expected, received_checksums)
        ), (f"expected={[round(x, 4) for x in expected]} " f"got={[round(x, 4) for x in received_checksums]}")
    finally:
        gc.collect()
        try:
            ttnn.close_mesh_device(mesh)
        except Exception as e:  # noqa: BLE001
            print(f"[rank {CONSUMER_RANK}] close_mesh_device: {type(e).__name__}: {e}", flush=True)


def test_rollout_queue() -> None:
    """End-to-end round-trip of ``N_BATCHES`` fake ``RolloutBatch``es."""
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()
    if _MPI_RANK == PRODUCER_RANK:
        _rank_producer_side()
    elif _MPI_RANK == CONSUMER_RANK:
        _rank_consumer_side()
    else:
        raise RuntimeError(f"Unexpected MPI rank {_MPI_RANK}; expected {PRODUCER_RANK} or {CONSUMER_RANK}.")

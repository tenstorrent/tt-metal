# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Two-rank tt-run test that runs :class:`ThreadedWeightBridge` (dict API)
and :class:`RolloutQueue` concurrently on separate duplicated MPI contexts.

Rank directions match the real fully-async trainer:

  * Rank 0 = TTML trainer.
      - ``ThreadedWeightBridge.sender``: ships fresh weight dicts to
        rank 1 via ``send_weights({"w0": x0, "w1": x1})``.
      - ``RolloutQueue.consumer``: reads incoming rollout batches.
  * Rank 1 = TTT rollout worker.
      - ``ThreadedWeightBridge.receiver``: reads weight dicts through
        ``with bridge.poll_weights() as dicts:`` between rounds.
      - ``RolloutQueue.producer``: pushes freshly generated batches.

Rank 0 main loop mirrors the training loop:

  1. Pop a rollout batch from ``RolloutQueue``.
  2. Run ``ttnn.add`` bursts on CQ0 (mock gradient work) on ``x0``
     (+1/add) and ``x1`` (+2/add).
  3. Every other round, ``send_weights({"w0": x0, "w1": x1})`` and then
     keep mutating x0 / x1 -- the pads are the freeze point.

Rank 1 main loop:

  1. Push a fake rollout batch through ``RolloutQueue``.
  2. ``with bridge.poll_weights() as dicts:`` non-blocking. If a fresh
     dict is there, sample both keys under the lock and record.

Verification: rank 1 ships two expected-summary lists at the end
(rollout checksums + observed weight first-elems per key). Rank 0
asserts both channels match.

Run via ``runner_bridge_and_queue.sh``.
"""

from __future__ import annotations

import os

import pytest

_WORLD_SIZE = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "0"))
if _WORLD_SIZE != 2:
    pytest.skip(
        "test_bridge_and_queue must run under tt-run with world_size == 2 (use runner_bridge_and_queue.sh).",
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
from utils.threaded_weight_bridge import ThreadedWeightBridge  # noqa: E402


# ---- knobs -------------------------------------------------------------------
TTML_RANK: int = 0
TTT_RANK: int = 1

MESH_SHAPE: tuple = (1, 1)
NUM_CQS: int = 2

PAD_SHAPE: tuple = (64, 64)
PAD_TTNN_DTYPE = ttnn.float32
PAD_TORCH_DTYPE = torch.float32

BATCH_B: int = 4
MAX_COMPLETION_LEN: int = 32
PROMPT_LEN: int = 6

N_ROUNDS: int = 6
QUEUE_CAPACITY: int = 2
ADDS_PER_ROUND: int = 200

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


def _fresh(mesh: "ttnn.MeshDevice", value: float) -> "ttnn.Tensor":
    return ttnn.from_torch(
        torch.full(PAD_SHAPE, value, dtype=PAD_TORCH_DTYPE),
        dtype=PAD_TTNN_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _build_batch(batch_id: int, weight_version: int) -> RolloutBatch:
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


def _drain_weight_bridge_once(bridge: "ThreadedWeightBridge", observed: List[dict]) -> bool:
    """Non-blockingly try to read ONE weight dict from the bridge.

    Returns True if a dict was consumed; False otherwise. Uses the
    ``poll_weights`` context manager, so the lock is held across the read
    when a dict is available.
    """
    with bridge.poll_weights() as dicts:
        if dicts is None:
            return False
        dev_dict = dicts[0]
        if not dev_dict:
            return False
        entry: dict = {}
        for k in sorted(dev_dict.keys()):
            entry[k] = float(ttnn.to_torch(dev_dict[k], cq_id=0)[0, 0])
    observed.append(entry)
    print(
        f"[rank {TTT_RANK}] observed fresh weight dict " + " ".join(f"{k}={v}" for k, v in entry.items()),
        flush=True,
    )
    return True


def _ttml_side() -> None:
    """Rank 0: weight bridge sender (dict API) + rollout queue consumer."""
    print(f"[rank {TTML_RANK}] TTML: opening mesh with num_command_queues={NUM_CQS}...", flush=True)
    mesh = _open_mesh()

    bridge = ThreadedWeightBridge.sender(peer_rank=TTT_RANK, mesh_device=mesh)
    q = RolloutQueue.consumer(peer_rank=TTT_RANK, capacity=QUEUE_CAPACITY)

    try:
        x0 = _fresh(mesh, 0.0)
        x1 = _fresh(mesh, 100.0)
        one = _fresh(mesh, 1.0)
        two = _fresh(mesh, 2.0)

        # Both connect calls are collective on their own duplicated context;
        # both ranks must call in the same order.
        bridge.connect()
        q.connect()
        print(f"[rank {TTML_RANK}] TTML: both channels connected", flush=True)

        received_rollout_checksums: List[float] = []
        pushed_weight_snapshots: List[dict] = []

        for r in range(N_ROUNDS):
            # 1) Pop a rollout batch.
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
                x0 = ttnn.add(x0, one)
                x1 = ttnn.add(x1, two)
            burst_ms = (time.perf_counter() - t0) * 1000.0

            # 3) Every other round, push the updated weight dict.
            if r % 2 == 0:
                first0 = float(ttnn.to_torch(x0, cq_id=0)[0, 0])
                first1 = float(ttnn.to_torch(x1, cq_id=0)[0, 0])
                pushed_weight_snapshots.append({"w0": first0, "w1": first1})
                bridge.send_weights({"w0": x0, "w1": x1})
                # Deliberate mutation after send_weights returns.
                for _ in range(ADDS_PER_ROUND):
                    x0 = ttnn.add(x0, one)
                    x1 = ttnn.add(x1, two)
                print(
                    f"[rank {TTML_RANK}] round {r}: pushed weight dict "
                    f"({ADDS_PER_ROUND} adds took {burst_ms:.1f}ms; then another "
                    f"{ADDS_PER_ROUND} post-push mutations; w0={first0}, w1={first1})",
                    flush=True,
                )
            else:
                print(
                    f"[rank {TTML_RANK}] round {r}: {ADDS_PER_ROUND} adds took {burst_ms:.1f}ms "
                    "(no push this round)",
                    flush=True,
                )

        print(f"[rank {TTML_RANK}] TTML: main loop done; closing bridge", flush=True)
        bridge.close()

        (n,) = struct.unpack("<I", ttnn.distributed_context_recv_bytes(4, TTT_RANK, _TAG_QUEUE_SUMMARY_LEN))
        expected_rollout = json.loads(
            ttnn.distributed_context_recv_bytes(int(n), TTT_RANK, _TAG_QUEUE_SUMMARY_BODY).decode()
        )
        (n,) = struct.unpack("<I", ttnn.distributed_context_recv_bytes(4, TTT_RANK, _TAG_BRIDGE_SUMMARY_LEN))
        expected_bridge = json.loads(
            ttnn.distributed_context_recv_bytes(int(n), TTT_RANK, _TAG_BRIDGE_SUMMARY_BODY).decode()
        )

        q.close()

        # Two independent PASS/FAIL asserts so the log shows which channel failed.
        assert expected_rollout == received_rollout_checksums, (
            f"rollout channel mismatch: expected={[round(x, 4) for x in expected_rollout]} "
            f"got={[round(x, 4) for x in received_rollout_checksums]}"
        )
        print(
            f"[rank {TTML_RANK}] rollout channel: PASS "
            f"expected={[round(x, 4) for x in expected_rollout]} "
            f"got={[round(x, 4) for x in received_rollout_checksums]}",
            flush=True,
        )
        assert (
            expected_bridge == pushed_weight_snapshots
        ), f"bridge channel mismatch: pushed={pushed_weight_snapshots} rank1_saw={expected_bridge}"
        print(
            f"[rank {TTML_RANK}] bridge channel:  PASS "
            f"pushed={pushed_weight_snapshots} rank1_saw={expected_bridge}",
            flush=True,
        )
        print(f"[rank {TTML_RANK}] [COMBINED PASS]", flush=True)
    finally:
        gc.collect()
        try:
            ttnn.close_mesh_device(mesh)
        except Exception as e:  # noqa: BLE001
            print(f"[rank {TTML_RANK}] close_mesh_device: {type(e).__name__}: {e}", flush=True)


def _ttt_side() -> None:
    """Rank 1: weight bridge receiver (dict API) + rollout queue producer."""
    print(f"[rank {TTT_RANK}] TTT: opening mesh with num_command_queues={NUM_CQS}...", flush=True)
    mesh = _open_mesh()

    bridge = ThreadedWeightBridge.receiver(peer_rank=TTML_RANK, mesh_device=mesh)
    q = RolloutQueue.producer(peer_rank=TTML_RANK, capacity=QUEUE_CAPACITY)

    try:
        bridge.connect()
        q.connect()
        print(f"[rank {TTT_RANK}] TTT: both channels connected", flush=True)

        pushed_rollout_checksums: List[float] = []
        observed_weight_snapshots: List[dict] = []

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

            # 2) Between rounds, non-blockingly poll for a fresh weight dict.
            _drain_weight_bridge_once(bridge, observed_weight_snapshots)

            time.sleep(0.05)

        # After the main loop, drain any remaining pending weight dicts.
        for _ in range(20):
            drained = _drain_weight_bridge_once(bridge, observed_weight_snapshots)
            if not drained:
                break
            time.sleep(0.02)

        print(f"[rank {TTT_RANK}] TTT: main loop done; closing queue", flush=True)
        q.close()

        body = json.dumps(pushed_rollout_checksums).encode()
        ttnn.distributed_context_send_bytes(struct.pack("<I", len(body)), TTML_RANK, _TAG_QUEUE_SUMMARY_LEN)
        ttnn.distributed_context_send_bytes(body, TTML_RANK, _TAG_QUEUE_SUMMARY_BODY)

        body = json.dumps(observed_weight_snapshots).encode()
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


def test_bridge_and_queue() -> None:
    """Combined end-to-end test: weight bridge and rollout queue running
    concurrently on separate duplicated MPI contexts."""
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()
    if _MPI_RANK == TTML_RANK:
        _ttml_side()
    elif _MPI_RANK == TTT_RANK:
        _ttt_side()
    else:
        raise RuntimeError(f"Unexpected MPI rank {_MPI_RANK}; expected 0 or 1.")

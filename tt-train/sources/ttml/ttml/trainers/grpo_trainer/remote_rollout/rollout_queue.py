# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Cross-rank queue of scored rollouts (``RolloutQueue``) on a private MPI communicator."""

from __future__ import annotations

import json
import queue
import struct
import threading
from typing import Any, Optional

import numpy as np

from ..distributed import exit_on_error
from ..grpo_trainer import RolloutBatch
from ..rollout_batch_source import ScoredRolloutBatch
from .weight_bridge import duplicate_distributed_context

# Tags are matched only within this queue's own communicator, so they cannot collide with other MPI traffic.
_TAG_LEN, _TAG_BODY = 0, 1

_STOP = object()


class RolloutQueue:
    """Single-producer single-consumer queue of ScoredRolloutBatches between two ranks over a private communicator.

    A transport thread per rank moves batches between a local queue of ``capacity`` items and the wire,
    so a full consumer blocks the producer's push.
    """

    def __init__(self, *, peer_rank: int, capacity: int, producer: bool) -> None:
        if capacity < 1:
            raise ValueError(f"RolloutQueue: capacity must be >= 1, got {capacity}")
        self._peer_rank = int(peer_rank)
        self._producer = producer
        self._local: queue.Queue = queue.Queue(maxsize=capacity)
        self._ctx: Any = None
        self._thread: Optional[threading.Thread] = None
        self._drained = False
        self._closed = False

    @classmethod
    def producer(cls, *, peer_rank: int, capacity: int) -> "RolloutQueue":
        return cls(peer_rank=peer_rank, capacity=capacity, producer=True)

    @classmethod
    def consumer(cls, *, peer_rank: int, capacity: int) -> "RolloutQueue":
        return cls(peer_rank=peer_rank, capacity=capacity, producer=False)

    def connect(self) -> None:
        self._ctx = duplicate_distributed_context()
        loop = self._producer_loop if self._producer else self._consumer_loop
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def push(self, scored: ScoredRolloutBatch) -> None:
        self._local.put(scored)

    def pop(self) -> Optional[ScoredRolloutBatch]:
        """Returns None once the producer has closed."""
        if self._drained:
            return None
        scored = self._local.get()
        if scored is None:
            self._drained = True
        return scored

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._producer:
            self._local.put(_STOP)
        else:
            while self.pop() is not None:
                pass
        if self._thread is not None:
            self._thread.join()

    def _producer_loop(self) -> None:
        with exit_on_error():
            while (scored := self._local.get()) is not _STOP:
                body = _encode(scored)
                self._ctx.send(struct.pack("<Q", len(body)), self._peer_rank, _TAG_LEN)
                self._ctx.send(body, self._peer_rank, _TAG_BODY)
            self._ctx.send(struct.pack("<Q", 0), self._peer_rank, _TAG_LEN)

    def _consumer_loop(self) -> None:
        with exit_on_error():
            while n := struct.unpack("<Q", self._ctx.recv(8, self._peer_rank, _TAG_LEN))[0]:
                self._local.put(_decode(self._ctx.recv(int(n), self._peer_rank, _TAG_BODY)))
            self._local.put(None)


def _encode(scored: ScoredRolloutBatch) -> bytes:
    r = scored.rollout
    return json.dumps(
        {
            "weight_version": r.weight_version,
            "prompts": r.prompts,
            "completions": r.completions,
            "logprobs": r.logprobs.tolist(),
            "rewards": scored.rewards.tolist(),
            "reward_components": {k: v.tolist() for k, v in scored.reward_components.items()},
            "generation_time_s": scored.generation_time_s,
            "reward_time_s": scored.reward_time_s,
        }
    ).encode("utf-8")


def _decode(body: bytes) -> ScoredRolloutBatch:
    d = json.loads(body)
    return ScoredRolloutBatch(
        rollout=RolloutBatch(
            weight_version=d["weight_version"],
            prompts=d["prompts"],
            completions=d["completions"],
            logprobs=np.asarray(d["logprobs"], dtype=np.float32),
        ),
        rewards=np.asarray(d["rewards"], dtype=np.float32),
        reward_components={k: np.asarray(v, dtype=np.float32) for k, v in d["reward_components"].items()},
        generation_time_s=d["generation_time_s"],
        reward_time_s=d["reward_time_s"],
    )

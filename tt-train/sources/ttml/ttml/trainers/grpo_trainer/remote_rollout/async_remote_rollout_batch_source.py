# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import Any, Iterator, Optional, Tuple

from ..grpo_ttml_model import weights_ref_hf_dict
from ..rollout_batch_source import RolloutBatchSource, ScoredRolloutBatch, validate_rollout_batch
from .async_rollout import AsyncRolloutClient


class AsyncRemoteRolloutBatchSource(RolloutBatchSource):
    """Yields the scored rollouts the rollout rank generates continuously, dropping stale ones.

    A batch generated with weight version v is dropped when the policy is more than max_staleness optimizer steps
    newer. The trainer pushes the policy as version 0 before training, then update_weights pushes it every
    ``weight_sync_every`` steps, as version ``step``. No push follows the last batch.
    """

    def __init__(
        self,
        *,
        client: AsyncRolloutClient,
        num_batches: int,
        num_generations: int,
        max_completion_length: int,
        weight_sync_every: int,
        max_staleness: int,
    ) -> None:
        self._client = client
        self._batches_left = int(num_batches)
        self._num_generations = num_generations
        self._max_completion_length = max_completion_length
        self._weight_sync_every = int(weight_sync_every)
        self._max_staleness = int(max_staleness)
        self._step = 0
        self._pushed_version: Optional[int] = None

    def __iter__(self) -> Iterator[ScoredRolloutBatch]:
        while self._batches_left > 0:
            scored, dropped = self._pop_fresh()
            self._batches_left -= 1
            validate_rollout_batch(
                scored.rollout,
                num_prompts=len(scored.rollout.prompts) // self._num_generations,
                num_generations=self._num_generations,
                max_completion_length=self._max_completion_length,
            )
            scored.rollouts_dropped = dropped
            yield scored

    def _pop_fresh(self) -> Tuple[ScoredRolloutBatch, int]:
        dropped = 0
        while True:
            scored = self._client.pop()
            if scored is None:
                raise RuntimeError("rollout rank closed the rollout queue")
            if self._step - scored.rollout.weight_version <= self._max_staleness:
                return scored, dropped
            dropped += 1

    def update_weights(self, model: Any, step: int) -> float:
        self._step = step
        if self._batches_left == 0:
            return 0.0
        if self._pushed_version is not None and step - self._pushed_version < self._weight_sync_every:
            return 0.0
        start = time.perf_counter()
        self._client.send_weights(weights_ref_hf_dict(model), version=step)
        self._pushed_version = step
        return time.perf_counter() - start

    def close(self) -> None:
        self._client.shutdown()

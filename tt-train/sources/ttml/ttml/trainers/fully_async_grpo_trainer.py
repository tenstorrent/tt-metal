# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Queue-driven, fully asynchronous GRPO training loop."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Sequence

from .grpo_trainer import GRPOTrainer


@dataclass(frozen=True)
class FullyAsyncRolloutBatch:
    """One lossless item consumed from the bounded rollout result queue."""

    prompts: List[List[int]]
    extra_columns: dict[str, list]
    completions: List[List[int]]
    behavior_version: int
    behavior_logprobs: List[List[float]]
    group_id: str

    def validate(self, num_generations: int) -> None:
        expected = len(self.prompts) * num_generations
        if len(self.completions) != expected:
            raise ValueError(
                f"rollout {self.group_id!r} has {len(self.completions)} completions; "
                f"expected {len(self.prompts)} prompts * {num_generations} generations = {expected}"
            )
        if len(self.behavior_logprobs) != expected:
            raise ValueError(
                f"rollout {self.group_id!r} has {len(self.behavior_logprobs)} logprob rows; expected {expected}"
            )
        for row, (tokens, logprobs) in enumerate(zip(self.completions, self.behavior_logprobs)):
            if len(tokens) != len(logprobs):
                raise ValueError(
                    f"rollout {self.group_id!r} row {row} has {len(tokens)} tokens "
                    f"but {len(logprobs)} behavior logprobs"
                )
        for name, values in self.extra_columns.items():
            if len(values) != len(self.prompts):
                raise ValueError(
                    f"rollout {self.group_id!r} column {name!r} has {len(values)} rows; "
                    f"expected {len(self.prompts)}"
                )


class FullyAsyncGRPOTrainer(GRPOTrainer):
    """Consume rollout batches while a remote worker independently produces.

    The completer owns the bounded prompt/result transport and the asynchronous
    weight lane.  It must expose ``start_async_rollouts``,
    ``await_async_rollout``, ``publish_weights`` and ``close_async_rollouts``.
    The initial policy publication is synchronized by ``start_async_rollouts``;
    subsequent publications return after a stable host snapshot has been
    handed to the background sender.

    Behavior-policy logprobs and versions are retained in
    :attr:`last_rollout_batch`.  They intentionally do not alter the loss yet;
    importance correction is a separately deferred piece of work.
    """

    _ASYNC_METHODS: Sequence[str] = (
        "start_async_rollouts",
        "await_async_rollout",
        "publish_weights",
        "close_async_rollouts",
    )

    def _require_async_completer(self) -> None:
        for name in self._ASYNC_METHODS:
            if not hasattr(self.completer, name):
                raise TypeError(
                    f"FullyAsyncGRPOTrainer requires the completer to expose {name}(); "
                    f"{type(self.completer).__name__} does not."
                )

    def train(self) -> None:
        self._require_async_completer()
        self._setup()
        for cb in self.callbacks:
            cb.on_train_begin(self)
        self.metrics = {"step": 0}
        self._reset_step_metrics()

        prompt_batches = list(self._iter_prompt_batches())
        self.completer.start_async_rollouts(prompt_batches, initial_version=0)
        self.last_rollout_batch: FullyAsyncRolloutBatch | None = None
        try:
            for _ in prompt_batches:
                wait_t0 = time.perf_counter()
                batch = self.completer.await_async_rollout()
                self.metrics["generation_wait_s"] = time.perf_counter() - wait_t0
                if not isinstance(batch, FullyAsyncRolloutBatch):
                    raise TypeError(
                        "await_async_rollout() must return FullyAsyncRolloutBatch, " f"got {type(batch).__name__}"
                    )
                batch.validate(self.config.num_generations)
                self.last_rollout_batch = batch
                self.metrics["behavior_version"] = batch.behavior_version

                prompts_x, cols_x = self._expand_prompts_and_columns(batch.prompts, batch.extra_columns)
                rewards_np = self._compute_rewards(prompts_x, batch.completions, cols_x)
                advantages_np = self._compute_advantages(rewards_np)

                for _iteration in range(self.config.num_iterations):
                    self._optimize(prompts_x, batch.completions, advantages_np)
                    self._apply_gradients()
                    self.metrics["step"] += 1
                    # Snapshot and enqueue theta_step before callbacks can do
                    # unrelated device work. Network transfer and rollout-side
                    # activation continue independently.
                    self.completer.publish_weights(self.metrics["step"])
                    self._publish_step_metrics()
                    self._maybe_checkpoint()
                    self._reset_step_metrics()
        finally:
            self.completer.close_async_rollouts()

        for cb in self.callbacks:
            cb.on_train_end(self)


__all__ = ["FullyAsyncGRPOTrainer", "FullyAsyncRolloutBatch"]

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Tuple

import numpy as np

from .grpo_trainer import RolloutBatch, RolloutSampler, _derive_reward_names, dispatch_reward


@dataclass
class ScoredRolloutBatch:
    rollout: RolloutBatch
    rewards: np.ndarray
    reward_components: Dict[str, np.ndarray]


class RolloutBatchSource(ABC):
    """Yields scored rollout batches to the trainer and receives its weight updates."""

    @abstractmethod
    def __iter__(self) -> Iterator[ScoredRolloutBatch]:
        ...

    @abstractmethod
    def update_weights(self, model: Any, step: int) -> None:
        ...

    def close(self) -> None:
        pass


def validate_rollout_batch(
    rollout: RolloutBatch, *, num_prompts: int, num_generations: int, max_completion_length: int
) -> None:
    B = num_prompts * num_generations
    if len(rollout.prompts) != B or len(rollout.completions) != B:
        raise RuntimeError(
            f"expected {B} rows, got {len(rollout.prompts)} prompts / {len(rollout.completions)} completions"
        )
    lp = rollout.logprobs
    if not isinstance(lp, np.ndarray) or lp.shape != (B, max_completion_length) or lp.dtype != np.float32:
        raise RuntimeError(
            f"logprobs must be a float32 array of shape [{B}, {max_completion_length}], "
            f"got {getattr(lp, 'dtype', type(lp))} {getattr(lp, 'shape', '')}"
        )
    for r, c in enumerate(rollout.completions):
        if len(c) > max_completion_length:
            raise RuntimeError(f"row {r}: completion length {len(c)} > max_completion_length {max_completion_length}")
        valid = lp[r, : len(c)]
        if not np.all(np.isfinite(valid)) or np.any(valid > 1e-3):
            raise RuntimeError(f"row {r}: invalid behavior log-probs (non-finite or > 0)")


def tokenize_prompts(dataset: Any, tokenizer: Any, num_prompts: int) -> Tuple[List[List[int]], Dict[str, list]]:
    dataset = dataset.select(range(num_prompts))
    prompts = [tokenizer.encode(row["prompt"]) for row in dataset]
    extra_columns = {k: list(dataset[k]) for k in dataset.column_names if k != "prompt"}
    return prompts, extra_columns


def score_rollout(
    rollout: RolloutBatch,
    prompts: List[List[int]],
    extra_columns: Dict[str, list],
    *,
    tokenizer: Any,
    reward_funcs: List[Callable[..., List[float]]],
    num_generations: int,
) -> ScoredRolloutBatch:
    g = num_generations
    if [list(p) for p in rollout.prompts] != [list(p) for p in prompts for _ in range(g)]:
        raise RuntimeError(
            "RolloutBatch.prompts must hold each prompt repeated num_generations times, in order "
            f"(got {len(rollout.prompts)} prompts for {len(prompts)} x {g})"
        )
    columns_x = {k: [v for v in col for _ in range(g)] for k, col in extra_columns.items()}
    prompt_strs = [tokenizer.decode(p) for p in rollout.prompts]
    completion_strs = [tokenizer.decode(c, skip_special_tokens=True) for c in rollout.completions]
    per_func = [
        np.array(dispatch_reward(fn, completion_strs, prompt_strs, columns_x), dtype=np.float32) for fn in reward_funcs
    ]
    return ScoredRolloutBatch(
        rollout=rollout,
        rewards=np.sum(per_func, axis=0).astype(np.float32),
        reward_components=dict(zip(_derive_reward_names(reward_funcs), per_func)),
    )


class InProcessRolloutBatchSource(RolloutBatchSource):
    def __init__(
        self,
        *,
        sampler: RolloutSampler,
        prompts: List[List[int]],
        extra_columns: Dict[str, list],
        batch_prompts: int,
        tokenizer: Any,
        reward_funcs: List[Callable[..., List[float]]],
        num_generations: int,
        max_completion_length: int,
    ) -> None:
        self.sampler = sampler
        self._prompts = prompts
        self._extra_columns = extra_columns
        self._batch_prompts = batch_prompts
        self._tokenizer = tokenizer
        self._reward_funcs = reward_funcs
        self._num_generations = num_generations
        self._max_completion_length = max_completion_length

    def __iter__(self) -> Iterator[ScoredRolloutBatch]:
        n = self._batch_prompts
        for start in range(0, len(self._prompts), n):
            prompts = [list(p) for p in self._prompts[start : start + n]]
            extra = {k: list(col[start : start + n]) for k, col in self._extra_columns.items()}
            rollout = self.sampler.generate(prompts)
            validate_rollout_batch(
                rollout,
                num_prompts=len(prompts),
                num_generations=self._num_generations,
                max_completion_length=self._max_completion_length,
            )
            yield score_rollout(
                rollout,
                prompts,
                extra,
                tokenizer=self._tokenizer,
                reward_funcs=self._reward_funcs,
                num_generations=self._num_generations,
            )

    def update_weights(self, model: Any, step: int) -> None:
        self.sampler.update_weights(None, version=step)


def build_rollout_batch_source(
    config: Any,
    *,
    sampler: RolloutSampler,
    prompts: List[List[int]],
    extra_columns: Dict[str, list],
    batch_prompts: int,
    tokenizer: Any,
    reward_funcs: List[Callable[..., List[float]]],
) -> RolloutBatchSource:
    if config.rollout_mode == "in_process":
        return InProcessRolloutBatchSource(
            sampler=sampler,
            prompts=prompts,
            extra_columns=extra_columns,
            batch_prompts=batch_prompts,
            tokenizer=tokenizer,
            reward_funcs=reward_funcs,
            num_generations=config.num_generations,
            max_completion_length=config.max_completion_length,
        )
    raise ValueError(f"unsupported rollout_mode {config.rollout_mode!r}")

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
from typing import Any, Callable, Dict, Iterator, List

from ..grpo_trainer import RolloutBatch
from ..grpo_ttml_model import weights_ref_hf_dict
from ..rollout_batch_source import (
    RolloutBatchSource,
    ScoredRolloutBatch,
    pad_logprobs,
    score_rollout,
    validate_rollout_batch,
)
from .mpi_rollout import MPIRolloutClient
from .weight_bridge import TTT_RANK, HostWeightBridge


class SyncRemoteRolloutBatchSource(RolloutBatchSource):
    """Generates each prompt batch on the rollout rank and scores it here.

    The rollout rank boots with the same weights at version 0; update_weights pushes the
    policy every ``weight_sync_every`` steps, as version ``step``.
    """

    def __init__(
        self,
        *,
        mesh_device: Any,
        prompts: List[List[int]],
        extra_columns: Dict[str, list],
        batch_prompts: int,
        tokenizer: Any,
        reward_funcs: List[Callable[..., List[float]]],
        num_generations: int,
        max_completion_length: int,
        weight_sync_every: int,
    ) -> None:
        self._prompts = prompts
        self._extra_columns = extra_columns
        self._batch_prompts = batch_prompts
        self._tokenizer = tokenizer
        self._reward_funcs = reward_funcs
        self._num_generations = num_generations
        self._max_completion_length = max_completion_length
        self._weight_sync_every = int(weight_sync_every)
        self._pushed_version = 0
        self._closed = False
        bridge = HostWeightBridge.init_sender(mesh=mesh_device, peer_rank=TTT_RANK)
        self._client = MPIRolloutClient(peer_rank=TTT_RANK, bridge=bridge)

    def __iter__(self) -> Iterator[ScoredRolloutBatch]:
        n, g = self._batch_prompts, self._num_generations
        for start in range(0, len(self._prompts), n):
            prompts = [list(p) for p in self._prompts[start : start + n]]
            extra = {k: list(col[start : start + n]) for k, col in self._extra_columns.items()}
            completions, token_logprobs, version = self._client.remote_generate(prompts)
            if version != self._pushed_version:
                raise RuntimeError(
                    f"rollout rank generated with weight_version {version}, expected {self._pushed_version}"
                )
            rollout = RolloutBatch(
                weight_version=version,
                prompts=[p for p in prompts for _ in range(g)],
                completions=completions,
                logprobs=pad_logprobs(completions, token_logprobs, self._max_completion_length),
            )
            validate_rollout_batch(
                rollout, num_prompts=len(prompts), num_generations=g, max_completion_length=self._max_completion_length
            )
            yield score_rollout(
                rollout, prompts, extra, tokenizer=self._tokenizer, reward_funcs=self._reward_funcs, num_generations=g
            )

    def update_weights(self, model: Any, step: int) -> None:
        if step % self._weight_sync_every != 0:
            return
        hf_dict = weights_ref_hf_dict(model)
        try:
            self._client.send_weights(hf_dict, version=step)
        finally:
            del hf_dict
            gc.collect()
        self._pushed_version = step

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._client.shutdown()

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Both ends of rollout_backend="ttt_remote_async" (``AsyncRolloutClient`` / ``AsyncRolloutServer``)."""

from __future__ import annotations

import itertools
import time
from typing import Any, Callable, Dict, List, Optional

from ..distributed import exit_on_error
from ..grpo_trainer import RolloutSampler
from ..rollout_batch_source import ScoredRolloutBatch, score_rollout
from .rollout_queue import RolloutQueue
from .threaded_weight_bridge import ThreadedWeightBridge
from .weight_bridge import TTML_RANK, TTT_RANK, _require_distributed_context


class AsyncRolloutClient:
    """Trainer-rank end of ttt_remote_async: pushes weights over a ThreadedWeightBridge and pops scored
    rollouts from a RolloutQueue. Connects both in the same order as AsyncRolloutServer."""

    def __init__(self, *, bridge: ThreadedWeightBridge, queue: RolloutQueue) -> None:
        local_rank = _require_distributed_context("AsyncRolloutClient")
        if local_rank != TTML_RANK:
            raise RuntimeError(f"AsyncRolloutClient must run on TTML_RANK={TTML_RANK} (got local rank {local_rank}).")
        self._bridge, self._queue = bridge, queue
        self._bridge.connect()
        self._queue.connect()
        self._shut_down = False

    def send_weights(self, hf_dict: dict, *, version: int) -> None:
        self._bridge.send_weights(hf_dict, version)

    def pop(self) -> Optional[ScoredRolloutBatch]:
        return self._queue.pop()

    def shutdown(self) -> None:
        if self._shut_down:
            return
        self._shut_down = True
        self._bridge.close()
        self._queue.close()


class AsyncRolloutServer:
    """Rollout-rank end of ttt_remote_async: generates and scores rollouts in a loop, without requests from the trainer.

    Unlike MPIRolloutServer, which generates only the prompts the trainer sends and installs weights only when
    asked, this server walks ``prompts`` itself in batches of ``batch_prompts``, wrapping around at the end. It
    scores each batch with ``reward_funcs`` and pushes the ScoredRolloutBatch to the RolloutQueue; a full queue
    blocks the push. Weights arrive in the background on the ThreadedWeightBridge while a batch generates. The
    newest received weights are installed into ``sampler`` before each batch, so every batch is generated with one
    weight version, which it carries. The server waits only for the first weights, and stops once the trainer
    closes the bridge. The caller owns ``sampler`` and closes it.
    """

    def __init__(
        self,
        *,
        bridge: ThreadedWeightBridge,
        queue: RolloutQueue,
        sampler: RolloutSampler,
        prompts: List[List[int]],
        extra_columns: Dict[str, list],
        batch_prompts: int,
        tokenizer: Any,
        reward_funcs: List[Callable[..., List[float]]],
        num_generations: int,
    ) -> None:
        local_rank = _require_distributed_context("AsyncRolloutServer")
        if local_rank != TTT_RANK:
            raise RuntimeError(f"AsyncRolloutServer must run on TTT_RANK={TTT_RANK} (got local rank {local_rank}).")
        self._bridge = bridge
        self._queue = queue
        self._sampler = sampler
        self._prompts = prompts
        self._extra_columns = extra_columns
        self._batch_prompts = batch_prompts
        self._tokenizer = tokenizer
        self._reward_funcs = reward_funcs
        self._num_generations = num_generations
        self._bridge.connect()
        self._queue.connect()

    def serve_forever(self) -> None:
        n = self._batch_prompts
        starts = itertools.cycle(range(0, len(self._prompts), n))
        with exit_on_error():
            while self._install_newest_weights():
                start = next(starts)
                prompts = self._prompts[start : start + n]
                extra = {k: col[start : start + n] for k, col in self._extra_columns.items()}
                t0 = time.perf_counter()
                rollout = self._sampler.generate(prompts)
                self._queue.push(
                    score_rollout(
                        rollout,
                        prompts,
                        extra,
                        tokenizer=self._tokenizer,
                        reward_funcs=self._reward_funcs,
                        num_generations=self._num_generations,
                        generation_time_s=time.perf_counter() - t0,
                    )
                )
            self._queue.close()
            self._bridge.close()

    def _install_newest_weights(self) -> bool:
        """Waits for the first weights; returns False once the trainer has closed the bridge."""
        with self._bridge.receive_weights(block=self._sampler.weight_version < 0) as received:
            if received is not None:
                version, weights = received
                self._sampler.update_weights(weights, version)
        return not self._bridge.peer_closed

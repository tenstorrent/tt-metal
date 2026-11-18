# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Both ends of rollout_backend="ttt_remote_async" (``AsyncRolloutClient`` / ``AsyncRolloutServer``)."""

from __future__ import annotations

import atexit
import itertools
import os
from typing import Optional

from ..distributed import exit_on_error
from ..rollout_batch_source import InProcessRolloutBatchSource, ScoredRolloutBatch
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
        atexit.register(self.shutdown)

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
    asked, this server cycles ``rollout_batch_source`` itself, wrapping around at the end. It pushes each
    ScoredRolloutBatch to the RolloutQueue; a full queue blocks the push. Weights arrive in the background on the
    ThreadedWeightBridge while a batch generates. The newest received weights are installed into
    ``rollout_batch_source.sampler`` before each batch, so every batch is generated with one weight version, which
    it carries. The server waits only for the first weights, and stops once the trainer closes the bridge. The
    caller owns ``rollout_batch_source.sampler`` and closes it.
    """

    def __init__(
        self, *, bridge: ThreadedWeightBridge, queue: RolloutQueue, rollout_batch_source: InProcessRolloutBatchSource
    ) -> None:
        local_rank = _require_distributed_context("AsyncRolloutServer")
        if local_rank != TTT_RANK:
            raise RuntimeError(f"AsyncRolloutServer must run on TTT_RANK={TTT_RANK} (got local rank {local_rank}).")
        self._bridge = bridge
        self._queue = queue
        self._rollout_batch_source = rollout_batch_source
        self._bridge.connect()
        self._queue.connect()
        # The peer waits on this rank until serve_forever returns; a normal exit before that hangs both ranks.
        atexit.register(os._exit, 1)

    def serve_forever(self) -> None:
        scored_batches = itertools.chain.from_iterable(itertools.repeat(self._rollout_batch_source))
        try:
            with exit_on_error():
                while self._install_newest_weights():
                    self._queue.push(next(scored_batches))
        finally:
            atexit.unregister(os._exit)
            self._queue.close()
            self._bridge.close()

    def _install_newest_weights(self) -> bool:
        """Waits for the first weights; returns False once the trainer has closed the bridge."""
        sampler = self._rollout_batch_source.sampler
        with self._bridge.receive_weights(block=sampler.weight_version < 0) as received:
            if received is not None:
                version, weights = received
                sampler.update_weights(weights, version)
        return not self._bridge.peer_closed

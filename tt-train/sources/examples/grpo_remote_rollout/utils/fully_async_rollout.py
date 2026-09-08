# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded prompt/result streaming and safe-boundary weight activation."""

from __future__ import annotations

import uuid
from threading import Thread
from typing import Any, Iterable

from ttml.trainers import FullyAsyncRolloutBatch

from .async_weight_bridge import AsyncHostWeightBridge
from .rollout_engine import PromptGroupLease, RolloutOutput, RolloutResult, UNBOUND_POLICY_VERSION
from .rollout_transport import RolloutTransportClosed, TrainerRolloutTransport, WorkerRolloutTransport


class FullyAsyncRolloutClient:
    """Trainer endpoint: feeder thread plus bounded result consumer."""

    def __init__(
        self,
        *,
        transport: TrainerRolloutTransport,
        weight_bridge: AsyncHostWeightBridge,
        weight_export,
        num_generations: int,
    ) -> None:
        self._transport = transport
        self._weight_bridge = weight_bridge
        self._weight_export = weight_export
        self._num_generations = int(num_generations)
        self._feeder: Thread | None = None
        self._feeder_error: BaseException | None = None
        self._started = False

    def start(self, prompt_batches: Iterable[tuple[list[list[int]], dict]], *, initial_version: int) -> None:
        if self._started:
            raise RuntimeError("fully async rollout client already started")
        self._started = True
        # The worker waits for this snapshot before accepting its first queued
        # lease, so no rollout can accidentally use dummy boot weights.
        self.publish_weights(initial_version)

        def feed() -> None:
            try:
                for batch_index, (prompts, extra_columns) in enumerate(prompt_batches):
                    expanded = [list(prompt) for prompt in prompts for _ in range(self._num_generations)]
                    payload = {
                        "prompts": [list(prompt) for prompt in prompts],
                        "extra_columns": {name: list(values) for name, values in extra_columns.items()},
                        "generation_prompts": expanded,
                    }
                    group_id = f"batch-{batch_index}"
                    self._transport.submit(
                        PromptGroupLease(
                            lease_id=str(uuid.uuid4()),
                            group_id=group_id,
                            behavior_version=UNBOUND_POLICY_VERSION,
                            payload=payload,
                        )
                    )
            except BaseException as error:
                self._feeder_error = error

        self._feeder = Thread(target=feed, name="fully-async-prompt-feeder", daemon=True)
        self._feeder.start()

    def receive(self) -> FullyAsyncRolloutBatch:
        if self._feeder_error is not None:
            raise RuntimeError("prompt feeder failed") from self._feeder_error
        result = self._transport.receive_result()
        payload = result.request_payload
        if not isinstance(payload, dict):
            raise RuntimeError(f"rollout {result.group_id!r} did not return its request payload")
        return FullyAsyncRolloutBatch(
            prompts=[[int(token) for token in row] for row in payload["prompts"]],
            extra_columns={name: list(values) for name, values in payload["extra_columns"].items()},
            completions=[list(row) for row in result.output.tokens],
            behavior_version=int(result.behavior_version),
            behavior_logprobs=[list(row) for row in result.output.logprobs],
            group_id=result.group_id,
        )

    def publish_weights(self, version: int) -> None:
        self._weight_bridge.publish(version, self._weight_export())

    def close(self) -> None:
        if self._feeder is not None:
            self._feeder.join()
        # Send the weight-lane close while the worker is still servicing its
        # receiver thread, then close the rollout lane.
        self._weight_bridge.close()
        self._transport.close()
        if self._feeder_error is not None:
            raise RuntimeError("prompt feeder failed") from self._feeder_error


class FullyAsyncRolloutWorker:
    """Inference endpoint that activates newest weights between generations."""

    def __init__(
        self,
        *,
        transport: WorkerRolloutTransport,
        weight_bridge: AsyncHostWeightBridge,
        worker: Any,
        max_new_tokens: int,
        enable_trace: bool = True,
        stop_at_eos: bool = True,
        engine_id: str = "ttt-rollout-0",
    ) -> None:
        self._transport = transport
        self._weight_bridge = weight_bridge
        self._worker = worker
        self._max_new_tokens = int(max_new_tokens)
        self._enable_trace = bool(enable_trace)
        self._stop_at_eos = bool(stop_at_eos)
        self._engine_id = engine_id
        self._active_version = -1

    def _activate(self, received) -> None:
        if received.version <= self._active_version:
            return
        per_submesh = self._weight_bridge.materialize(received)
        self._worker.update_weights(per_submesh)
        self._active_version = received.version

    def serve_forever(self) -> None:
        # Initial SFT snapshot is mandatory and ordered before all generation.
        self._activate(self._weight_bridge.wait())
        try:
            while True:
                try:
                    lease = self._transport.receive()
                except RolloutTransportClosed:
                    return
                if not isinstance(lease, PromptGroupLease):
                    raise TypeError(f"fully async worker only accepts prompt leases, got {type(lease).__name__}")
                newest = self._weight_bridge.poll()
                if newest is not None:
                    self._activate(newest)
                payload = lease.payload
                prompts = payload["generation_prompts"] if isinstance(payload, dict) else payload
                output = self._worker.generate(
                    prompts,
                    max_new_tokens=self._max_new_tokens,
                    enable_trace=self._enable_trace,
                    stop_at_eos=self._stop_at_eos,
                )
                if not isinstance(output, RolloutOutput):
                    raise TypeError("generation worker must return RolloutOutput with behavior logprobs")
                self._transport.publish(
                    RolloutResult(
                        engine_id=self._engine_id,
                        lease_id=lease.lease_id,
                        group_id=lease.group_id,
                        attempt_id=lease.attempt_id,
                        behavior_version=self._active_version,
                        output=output,
                        request_payload=lease.payload,
                    )
                )
        finally:
            self._transport.close()


__all__ = ["FullyAsyncRolloutClient", "FullyAsyncRolloutWorker"]

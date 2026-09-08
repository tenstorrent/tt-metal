# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TT-Transformer implementation of the backend-neutral rollout lifecycle.

This adapter deliberately contains no TTNN imports.  It wraps an already
constructed ``TttGenerationWorker`` and receiver-side ``WeightBridge`` so its
control flow can be tested on a host without a TT device.

Generation and weight receipt are synchronous calls, but the base class never
holds its state lock while invoking them.  A service may therefore run control
and weight-transfer commands on separate threads: the final old-policy rollout
can overlap receipt into the bridge's inactive storage, while activation remains
gated on both operations completing.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol, Sequence

from .rollout_engine import EventSink, PolicyVersion, PromptGroupLease, RolloutEngine, RolloutOutput


class _GenerationWorker(Protocol):
    models: Sequence[Any]

    def generate(
        self,
        prompts: Any,
        *,
        max_new_tokens: int,
        enable_trace: bool,
        stop_at_eos: bool,
    ) -> RolloutOutput:
        ...

    def update_weights(self, per_submesh: list[dict[str, Any]]) -> None:
        ...


class _ReceiverWeightBridge(Protocol):
    def receive_weights(self) -> list[dict[str, Any]]:
        ...

    def barrier(self) -> None:
        ...


@dataclass(frozen=True)
class _StagedWeights:
    """The one version allowed to occupy receiver-side staging."""

    version: PolicyVersion
    per_submesh: list[dict[str, Any]]


class TttRolloutEngine(RolloutEngine):
    """Run the common lifecycle using ``TttGenerationWorker`` operations.

    ``PromptGroupLease.payload`` is passed directly to ``worker.generate`` and
    is therefore expected to be a batch of token-ID prompts.  ``source`` on
    ``stage_weights`` is intentionally unused: a ``WeightBridge`` is already
    bound to its peer when it is constructed and connected.

    Only one private staged-weight reference is needed because
    :class:`RolloutEngine` rejects a second stage request until the first target
    activates or fails.  No separate staging-buffer abstraction is introduced.

    MULTI-WORKER EXTENSION: construct one ``TttRolloutEngine`` (and one worker /
    receiver bridge endpoint) per independently schedulable rollout worker.
    The coordinator routes leases and version commands by ``engine_id``; this
    adapter and its single-worker state transitions do not change.
    """

    def __init__(
        self,
        *,
        engine_id: str,
        active_version: PolicyVersion,
        worker: _GenerationWorker,
        weight_bridge: _ReceiverWeightBridge,
        max_new_tokens: int = 128,
        enable_trace: bool = True,
        stop_at_eos: bool = True,
        event_sink: EventSink | None = None,
    ) -> None:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        target_count = len(worker.models)
        if target_count == 0:
            raise ValueError("worker must own at least one model target")

        self._worker = worker
        self._weight_bridge = weight_bridge
        self._target_count = target_count
        self._max_new_tokens = max_new_tokens
        self._enable_trace = enable_trace
        self._stop_at_eos = stop_at_eos
        self._staged: _StagedWeights | None = None
        self._staged_lock = RLock()

        super().__init__(engine_id=engine_id, active_version=active_version, event_sink=event_sink)

    def _start_rollout_action(self, lease: PromptGroupLease) -> None:
        prompts = lease.payload
        if isinstance(prompts, dict) and "generation_prompts" in prompts:
            prompts = prompts["generation_prompts"]
        payload = self._worker.generate(
            prompts,
            max_new_tokens=self._max_new_tokens,
            enable_trace=self._enable_trace,
            stop_at_eos=self._stop_at_eos,
        )
        if not isinstance(payload, RolloutOutput):
            raise TypeError("worker.generate must return RolloutOutput containing host tokens and logprobs")
        self.rollout_completed(lease.lease_id, payload)

    def _stage_weights_action(self, version: PolicyVersion, source: Any) -> None:
        del source  # The receiver bridge is already bound to its sender.

        per_submesh = self._weight_bridge.receive_weights()
        # Pair every completed receive with the sender-side fence before
        # publishing WeightsStaged.  The sender may then release/reuse sources.
        self._weight_bridge.barrier()
        self._validate_received_weights(per_submesh)

        with self._staged_lock:
            if self._staged is not None:
                # The base lifecycle prevents this; keep the resource invariant
                # local as protection against future adapter-only changes.
                raise RuntimeError(f"staged version {self._staged.version} has not been consumed")
            self._staged = _StagedWeights(version, per_submesh)

        self.weights_staged(version)

    def _activate_staged_action(self, version: PolicyVersion) -> None:
        with self._staged_lock:
            staged = self._require_staged(version)
            # Keep the reference pinned until all device-local copies have been
            # enqueued/completed by the worker implementation.
            self._worker.update_weights(staged.per_submesh)
            self._staged = None

        self.activation_completed(version)

    def _discard_staged_action(self, version: PolicyVersion) -> None:
        with self._staged_lock:
            if self._staged is not None and self._staged.version == version:
                self._staged = None

    def _validate_received_weights(self, per_submesh: list[dict[str, Any]]) -> None:
        if len(per_submesh) != self._target_count:
            raise ValueError(f"received weights for {len(per_submesh)} targets; worker expects {self._target_count}")
        for target_index, weights in enumerate(per_submesh):
            if not isinstance(weights, dict) or not weights:
                raise ValueError(f"received empty or invalid weight dictionary for target {target_index}")

    def _require_staged(self, version: PolicyVersion) -> _StagedWeights:
        if self._staged is None or self._staged.version != version:
            actual = self._staged.version if self._staged is not None else None
            raise RuntimeError(f"cannot activate version {version}; adapter holds staged version {actual}")
        return self._staged


__all__ = ["TttRolloutEngine"]

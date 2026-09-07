# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Trainer-side coordination for one asynchronously executing rollout worker."""

from __future__ import annotations

from collections import deque
from threading import RLock
from time import monotonic
from typing import Any, Protocol

from .rollout_engine import (
    EngineEvent,
    PolicyActivated,
    PolicyVersion,
    PromptGroupLease,
    ResultReady,
    RolloutResult,
    WeightsStaged,
)
from .rollout_transport import RemoteRolloutError, TrainerRolloutTransport


class _SenderWeightBridge(Protocol):
    def send_weights(self, weights: Any) -> None:
        ...

    def barrier(self) -> None:
        ...


class SingleWorkerRolloutCoordinator:
    """Own trainer-side leases and policy cutovers for one rollout engine.

    Result and lifecycle events share one ordered lane.  Waiting for a policy
    activation therefore also drains completed old-policy rollouts into a
    local ready queue; no result is lost merely because it races a cutover.

    This first implementation assumes one orchestration thread and one rollout
    worker.  MULTI-WORKER EXTENSION: lift the fields in this class into an
    ``engine_id -> worker state`` map, broadcast cutovers to each endpoint, and
    route events through the same ``engine_id`` validation performed here.
    The transport messages and engine state machine do not need to change.
    """

    def __init__(
        self,
        *,
        engine_id: str,
        active_version: PolicyVersion,
        transport: TrainerRolloutTransport,
        weight_bridge: _SenderWeightBridge,
    ) -> None:
        if not engine_id:
            raise ValueError("engine_id must be non-empty")
        if active_version < 0:
            raise ValueError("active_version must be non-negative")
        self._engine_id = engine_id
        self._active_version = active_version
        self._transport = transport
        self._weight_bridge = weight_bridge
        self._target_version: PolicyVersion | None = None
        self._staged_version: PolicyVersion | None = None
        self._outstanding: dict[str, PromptGroupLease] = {}
        self._completed_lease_ids: set[str] = set()
        self._ready_results: deque[RolloutResult] = deque()
        self._failure: Exception | None = None
        self._lock = RLock()

    @property
    def engine_id(self) -> str:
        return self._engine_id

    @property
    def active_version(self) -> PolicyVersion:
        with self._lock:
            return self._active_version

    @property
    def outstanding_count(self) -> int:
        with self._lock:
            return len(self._outstanding)

    def submit(self, lease: PromptGroupLease, *, timeout: float | None = None) -> None:
        """Validate and submit a lease against the currently active policy."""
        with self._lock:
            self._raise_if_failed()
            if self._target_version is not None:
                raise RuntimeError(f"cannot submit a lease during cutover to version {self._target_version}")
            if lease.behavior_version != self._active_version:
                raise ValueError(
                    f"lease behavior version {lease.behavior_version} does not match "
                    f"coordinator active version {self._active_version}"
                )
            if lease.lease_id in self._outstanding or lease.lease_id in self._completed_lease_ids:
                raise ValueError(f"lease_id {lease.lease_id!r} has already been submitted")
            self._outstanding[lease.lease_id] = lease

        try:
            self._transport.submit(lease, timeout=timeout)
        except Exception:
            with self._lock:
                self._outstanding.pop(lease.lease_id, None)
            raise

    def begin_policy_cutover(
        self,
        version: PolicyVersion,
        weights: Any,
        *,
        timeout: float | None = None,
    ) -> None:
        """Quiesce, establish staging intent, and transfer ``version``.

        This returns after the weight bridge's sender-side fence.  Activation
        is a separate device-local operation; call :meth:`await_policy_activation`
        before issuing leases labeled with the new behavior version.
        """
        deadline = self._deadline(timeout)
        with self._lock:
            self._raise_if_failed()
            if self._target_version is not None:
                raise RuntimeError(f"cutover to version {self._target_version} is already in progress")
            if version <= self._active_version:
                raise ValueError(f"target version {version} must be newer than active version {self._active_version}")
            self._target_version = version
            self._staged_version = None

        try:
            self._transport.quiesce(version, timeout=self._remaining(deadline))
            # MPI waits until this command is on the wire.  The receiver can
            # then enter receive_weights before the sender bridge starts.
            self._transport.request_weight_stage(version, timeout=self._remaining(deadline))
            self._weight_bridge.send_weights(weights)
            self._weight_bridge.barrier()
        except Exception as error:
            self._record_failure(error)
            raise

    def await_policy_activation(
        self,
        version: PolicyVersion,
        *,
        timeout: float | None = None,
    ) -> None:
        """Wait for the matching activation acknowledgement, buffering results."""
        deadline = self._deadline(timeout)
        with self._lock:
            self._raise_if_failed()
            if self._target_version is None:
                if version == self._active_version:
                    return
                raise RuntimeError(f"no cutover to version {version} is in progress")
            if version != self._target_version:
                raise RuntimeError(f"waiting for version {version}, but cutover targets version {self._target_version}")

        while True:
            self._receive_and_record(timeout=self._remaining(deadline))
            with self._lock:
                self._raise_if_failed()
                if self._active_version == version and self._target_version is None:
                    return

    def activate_policy(
        self,
        version: PolicyVersion,
        weights: Any,
        *,
        timeout: float | None = None,
    ) -> None:
        """Convenience wrapper for beginning and awaiting one policy cutover."""
        deadline = self._deadline(timeout)
        self.begin_policy_cutover(version, weights, timeout=self._remaining(deadline))
        self.await_policy_activation(version, timeout=self._remaining(deadline))

    def receive_event(self, *, timeout: float | None = None) -> EngineEvent:
        """Consume and validate one event, updating coordinator bookkeeping."""
        return self._receive_and_record(timeout=timeout)

    def receive_result(self, *, timeout: float | None = None) -> RolloutResult:
        """Return a validated result, consuming lifecycle events as necessary."""
        deadline = self._deadline(timeout)
        while True:
            with self._lock:
                self._raise_if_failed()
                if self._ready_results:
                    return self._ready_results.popleft()
            self._receive_and_record(timeout=self._remaining(deadline))

    def close(self) -> None:
        self._transport.close()

    def _receive_and_record(self, *, timeout: float | None) -> EngineEvent:
        try:
            event = self._transport.receive_event(timeout=timeout)
        except RemoteRolloutError as error:
            self._record_failure(error)
            raise
        try:
            self._record_event(event)
        except Exception as error:
            self._record_failure(error)
            raise
        return event

    def _record_event(self, event: EngineEvent) -> None:
        with self._lock:
            event_engine_id = event.result.engine_id if isinstance(event, ResultReady) else event.engine_id
            if event_engine_id != self._engine_id:
                raise RuntimeError(f"received event from engine {event_engine_id}; expected {self._engine_id}")
            if isinstance(event, ResultReady):
                result = event.result
                lease = self._outstanding.get(result.lease_id)
                if lease is None:
                    raise RuntimeError(f"result for unknown or completed lease {result.lease_id!r}")
                expected = (lease.group_id, lease.behavior_version, lease.attempt_id)
                actual = (result.group_id, result.behavior_version, result.attempt_id)
                if actual != expected:
                    raise RuntimeError(
                        f"result for lease {result.lease_id!r} does not match submitted lease "
                        f"(expected {expected}, got {actual})"
                    )
                del self._outstanding[result.lease_id]
                self._completed_lease_ids.add(result.lease_id)
                self._ready_results.append(result)
            elif isinstance(event, WeightsStaged):
                if event.version != self._target_version:
                    raise RuntimeError(
                        f"weights-staged version {event.version} does not match cutover target {self._target_version}"
                    )
                if self._staged_version is not None:
                    raise RuntimeError(f"duplicate weights-staged event for version {event.version}")
                self._staged_version = event.version
            elif isinstance(event, PolicyActivated):
                if event.version != self._target_version:
                    raise RuntimeError(
                        f"activated version {event.version} does not match cutover target {self._target_version}"
                    )
                if self._staged_version != event.version:
                    raise RuntimeError(
                        f"policy version {event.version} activated before weights-staged acknowledgement"
                    )
                self._active_version = event.version
                self._target_version = None
                self._staged_version = None
            else:
                # EngineFailed is translated to RemoteRolloutError by every
                # trainer transport before it reaches this method.
                raise TypeError(f"unsupported rollout event {type(event).__name__}")

    def _record_failure(self, error: Exception) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = error

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise self._failure

    @staticmethod
    def _deadline(timeout: float | None) -> float | None:
        return None if timeout is None else monotonic() + timeout

    @staticmethod
    def _remaining(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - monotonic())


__all__ = ["SingleWorkerRolloutCoordinator"]

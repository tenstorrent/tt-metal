# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host-side queues and contracts for asynchronous rollout communication.

The transport boundary contains ordinary Python values, never device tensors.
TT backends are responsible for completing device-to-host reads before they
publish a :class:`RolloutResult`.  This keeps queueing, backpressure, and wire
protocol code independent of TTNN and makes it device-free to test.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import deque
from queue import Empty, Full
from threading import Condition
from time import monotonic
from typing import Generic, TypeVar

from .rollout_engine import EngineFailed, PolicyVersion, PromptGroupLease, RolloutResult


class RolloutTransportClosed(RuntimeError):
    """The requested queue endpoint has been closed."""


class RemoteRolloutError(RuntimeError):
    """A rollout engine reported a terminal failure."""

    def __init__(self, failure: EngineFailed) -> None:
        super().__init__(f"engine {failure.engine_id} failed during {failure.operation}: {failure.message}")
        self.failure = failure


@dataclass(frozen=True)
class QuiescePolicy:
    target_version: PolicyVersion


@dataclass(frozen=True)
class StagePolicyWeights:
    version: PolicyVersion


RolloutCommand = PromptGroupLease | QuiescePolicy | StagePolicyWeights


class TrainerRolloutTransport(ABC):
    """Trainer-side half of a bounded prompt/result transport."""

    @abstractmethod
    def submit(self, lease: PromptGroupLease, *, timeout: float | None = None) -> None:
        """Submit a lease, applying backpressure when capacity is exhausted."""

    @abstractmethod
    def receive_result(self, *, timeout: float | None = None) -> RolloutResult:
        """Return the next completed rollout."""

    @abstractmethod
    def quiesce(self, target_version: PolicyVersion, *, timeout: float | None = None) -> None:
        """Stop issuance of old-policy work before a policy cutover."""

    @abstractmethod
    def request_weight_stage(self, version: PolicyVersion, *, timeout: float | None = None) -> None:
        """Deliver staging intent before the caller starts ``WeightBridge.send_weights``."""

    @abstractmethod
    def close(self) -> None:
        """Refuse future submissions and eventually stop the worker endpoint."""


class WorkerRolloutTransport(ABC):
    """Rollout-worker-side half of a bounded prompt/result transport."""

    @abstractmethod
    def receive(self, *, timeout: float | None = None) -> RolloutCommand:
        """Return the next lease or policy-lifecycle command."""

    @abstractmethod
    def publish(self, result: RolloutResult, *, timeout: float | None = None) -> None:
        """Publish a completed rollout, applying result-side backpressure."""

    @abstractmethod
    def publish_failure(self, failure: EngineFailed) -> None:
        """Publish a terminal engine failure and wake blocked consumers."""

    @abstractmethod
    def close(self) -> None:
        """Finish result delivery after all accepted leases are handled."""


@dataclass(frozen=True)
class InMemoryRolloutTransports:
    """Paired endpoints used by unit tests and same-process coordinators."""

    trainer: TrainerRolloutTransport
    worker: WorkerRolloutTransport


_T = TypeVar("_T")


class _ClosableQueue(Generic[_T]):
    """Small bounded queue whose close cannot be lost when it is full."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._items: deque[_T] = deque()
        self._closed = False
        self._condition = Condition()

    def put(self, item: _T, timeout: float | None) -> None:
        deadline = None if timeout is None else monotonic() + timeout
        with self._condition:
            while len(self._items) >= self._capacity:
                if self._closed:
                    raise RolloutTransportClosed("rollout transport is closed")
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    raise Full
                self._condition.wait(remaining)
            if self._closed:
                raise RolloutTransportClosed("rollout transport is closed")
            self._items.append(item)
            self._condition.notify_all()

    def get(self, timeout: float | None) -> _T:
        deadline = None if timeout is None else monotonic() + timeout
        with self._condition:
            while not self._items:
                if self._closed:
                    raise RolloutTransportClosed("rollout transport is closed")
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    raise Empty
                self._condition.wait(remaining)
            item = self._items.popleft()
            self._condition.notify_all()
            return item

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class _SharedQueues:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.requests = _ClosableQueue[RolloutCommand](capacity)
        self.results = _ClosableQueue[RolloutResult | EngineFailed](capacity)

    def close(self) -> None:
        self.requests.close()
        self.results.close()


class _InMemoryTrainerTransport(TrainerRolloutTransport):
    def __init__(self, shared: _SharedQueues) -> None:
        self._shared = shared

    def submit(self, lease: PromptGroupLease, *, timeout: float | None = None) -> None:
        self._shared.requests.put(lease, timeout=timeout)

    def receive_result(self, *, timeout: float | None = None) -> RolloutResult:
        value = self._shared.results.get(timeout)
        if isinstance(value, EngineFailed):
            raise RemoteRolloutError(value)
        return value

    def quiesce(self, target_version: PolicyVersion, *, timeout: float | None = None) -> None:
        self._shared.requests.put(QuiescePolicy(target_version), timeout)

    def request_weight_stage(self, version: PolicyVersion, *, timeout: float | None = None) -> None:
        self._shared.requests.put(StagePolicyWeights(version), timeout)

    def close(self) -> None:
        self._shared.close()


class _InMemoryWorkerTransport(WorkerRolloutTransport):
    def __init__(self, shared: _SharedQueues) -> None:
        self._shared = shared

    def receive(self, *, timeout: float | None = None) -> RolloutCommand:
        return self._shared.requests.get(timeout)

    def publish(self, result: RolloutResult, *, timeout: float | None = None) -> None:
        self._shared.results.put(result, timeout=timeout)

    def publish_failure(self, failure: EngineFailed) -> None:
        self._shared.results.put(failure, timeout=None)

    def close(self) -> None:
        self._shared.close()


def create_in_memory_rollout_transports(*, capacity: int = 1) -> InMemoryRolloutTransports:
    """Create a bounded transport pair with the same semantics as MPI queues.

    MULTI-WORKER EXTENSION: create one pair per worker and route by
    ``engine_id``.  The endpoint contracts and message types do not change.
    """

    shared = _SharedQueues(capacity)
    return InMemoryRolloutTransports(
        trainer=_InMemoryTrainerTransport(shared),
        worker=_InMemoryWorkerTransport(shared),
    )


__all__ = [
    "InMemoryRolloutTransports",
    "QuiescePolicy",
    "RemoteRolloutError",
    "RolloutCommand",
    "RolloutTransportClosed",
    "StagePolicyWeights",
    "TrainerRolloutTransport",
    "WorkerRolloutTransport",
    "create_in_memory_rollout_transports",
]

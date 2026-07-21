# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Blocking and asynchronous device-to-host output read lifecycle."""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from typing import Any

import torch

import ttnn


@dataclass(frozen=True)
class PendingRead:
    """One retained asynchronous host destination and its completion events."""

    value: Any
    events: tuple[Any, ...]
    sequence: int
    _owner: Any = field(repr=False, compare=False)
    _completed: bool = field(default=False, repr=False, compare=False)

    @property
    def completed(self) -> bool:
        return self._completed


class OutputReader:
    """Own pending output-read destinations/events until completion or drain.

    Blocking versus asynchronous behavior is selected on each call. There is
    intentionally no separate static configuration object.
    """

    def __init__(self, mesh_device: Any):
        self.mesh_device = mesh_device
        self._sequences = itertools.count()
        self._pending: dict[int, PendingRead] = {}
        self._pending_by_value_id: dict[int, int] = {}
        self._owner = object()
        self._lock = threading.Lock()

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def pending_reads(self) -> tuple[PendingRead, ...]:
        with self._lock:
            return tuple(self._pending.values())

    def read(self, value: Any, *, blocking: bool = True) -> Any | PendingRead:
        """Read a nested output payload to host.

        Blocking calls return the completed host payload directly. Async calls
        return a ``PendingRead`` whose ``value`` and ``events`` can be adapted to
        the existing vLLM tuple contract while this reader retains ownership.
        """

        retained_destinations: list[Any] = []
        try:
            host_value, submitted = _read_to_host(
                value,
                blocking=blocking,
                retained_destinations=retained_destinations,
            )
        except BaseException:
            if not blocking:
                _synchronize_after_failed_submission(self.mesh_device)
            raise
        if blocking:
            return host_value

        sequence = next(self._sequences)
        if not submitted:
            return PendingRead(value=host_value, events=(), sequence=sequence, _owner=self._owner, _completed=True)

        record_event = getattr(ttnn, "record_event", None)
        event_synchronize = getattr(ttnn, "event_synchronize", None)
        if not callable(record_event) or not callable(event_synchronize):
            _synchronize_after_failed_submission(self.mesh_device)
            raise RuntimeError("Asynchronous output reads require ttnn.record_event and ttnn.event_synchronize")
        try:
            event = record_event(self.mesh_device, 0)
        except BaseException:
            _synchronize_after_failed_submission(self.mesh_device)
            raise

        pending = PendingRead(value=host_value, events=(event,), sequence=sequence, _owner=self._owner)
        with self._lock:
            self._pending[sequence] = pending
            self._pending_by_value_id[id(host_value)] = sequence
        return pending

    def submit(self, value: Any) -> PendingRead:
        """Submit an asynchronous read and retain its resources."""

        result = self.read(value, blocking=False)
        assert isinstance(result, PendingRead)
        return result

    def complete(self, pending_or_value: PendingRead | Any) -> Any:
        """Synchronize and retire a pending read, returning its host payload.

        Passing the exact unwrapped ``PendingRead.value`` supports compatibility
        facades that return ``(host_value, events)`` to external schedulers.
        Completion is idempotent.
        """

        if isinstance(pending_or_value, PendingRead):
            candidate = pending_or_value
            if candidate._owner is not self._owner:
                raise ValueError("PendingRead is not owned by this OutputReader")
            sequence = candidate.sequence
        else:
            with self._lock:
                sequence = self._pending_by_value_id.get(id(pending_or_value))
            if sequence is None:
                return pending_or_value
            candidate = None

        with self._lock:
            pending = self._pending.get(sequence)
        if pending is None:
            if candidate is not None:
                if candidate.completed:
                    return candidate.value
                raise ValueError("PendingRead is not active in this OutputReader")
            return pending_or_value
        if candidate is not None and pending is not candidate:
            raise ValueError("PendingRead is not owned by this OutputReader")

        synchronize = getattr(ttnn, "event_synchronize", None)
        if pending.events and not callable(synchronize):
            raise RuntimeError("Pending output reads require ttnn.event_synchronize")
        for event in pending.events:
            synchronize(event)

        with self._lock:
            current = self._pending.pop(sequence, None)
            if current is not None:
                self._pending_by_value_id.pop(id(current.value), None)
                object.__setattr__(current, "_completed", True)
        return pending.value

    def drain(self) -> None:
        """Complete every pending read; repeated drains are no-ops."""

        with self._lock:
            pending_reads = tuple(self._pending.values())
        failures = []
        for pending in pending_reads:
            try:
                self.complete(pending)
            except BaseException as error:
                failures.append(error)
        if failures:
            raise RuntimeError(f"Failed to drain {len(failures)} pending output read(s)") from failures[0]


def _read_to_host(
    value: Any,
    *,
    blocking: bool,
    retained_destinations: list[Any],
) -> tuple[Any, bool]:
    if value is None:
        return None, False
    if isinstance(value, tuple):
        converted = [
            _read_to_host(item, blocking=blocking, retained_destinations=retained_destinations) for item in value
        ]
        return tuple(item for item, _ in converted), any(submitted for _, submitted in converted)
    if isinstance(value, list):
        converted = [
            _read_to_host(item, blocking=blocking, retained_destinations=retained_destinations) for item in value
        ]
        return [item for item, _ in converted], any(submitted for _, submitted in converted)
    if isinstance(value, dict):
        converted = {
            key: _read_to_host(item, blocking=blocking, retained_destinations=retained_destinations)
            for key, item in value.items()
        }
        return (
            {key: item for key, (item, _) in converted.items()},
            any(submitted for _, submitted in converted.values()),
        )
    if isinstance(value, torch.Tensor):
        return value.cpu(), False
    if isinstance(value, ttnn.Tensor):
        if value.storage_type() == ttnn.StorageType.HOST:
            return value, False
        host_value = value.cpu(blocking=blocking)
        if not blocking:
            retained_destinations.append(host_value)
        return host_value, not blocking

    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        try:
            host_value = cpu(blocking=blocking)
        except TypeError:
            return cpu(), False
        if not blocking:
            retained_destinations.append(host_value)
        return host_value, not blocking
    return value, False


def _synchronize_after_failed_submission(mesh_device: Any) -> None:
    synchronize = getattr(ttnn, "synchronize_device", None)
    if callable(synchronize):
        synchronize(mesh_device)

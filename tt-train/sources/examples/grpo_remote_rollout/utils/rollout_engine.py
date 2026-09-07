# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral rollout-engine lifecycle.

``RolloutEngine`` owns policy-version and prompt-lease state.  Concrete engines
only implement the side effects requested by the lifecycle: start generation,
receive weights into an inactive buffer, copy staged weights into the active
model, discard staging, and shut down.

The side-effect methods *initiate* work.  They must report completion through
the corresponding final base-class method (``rollout_completed``,
``weights_staged``, or ``activation_completed``).  This permits generation and
weight staging to run concurrently without allowing backend callbacks to mutate
state directly.

The first integration supports one in-flight prompt group on one rollout
worker.  MULTI-WORKER EXTENSION: instantiate one engine per independently
schedulable worker and have the coordinator route commands/events by
``engine_id``.  This class, its transitions, and its event types require no
change; only the coordinator's single engine reference becomes an engine map.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from threading import RLock
from typing import Any, Callable, Final, Optional, Sequence, final


PolicyVersion = int
EventSink = Callable[["EngineEvent"], None]


class EngineState(Enum):
    READY = auto()
    GENERATING = auto()
    DRAINING = auto()
    WAITING_FOR_STAGE = auto()
    ACTIVATING = auto()
    FAILED = auto()
    STOPPED = auto()


class InvalidTransitionError(RuntimeError):
    """A command or completion event is invalid for the current lifecycle."""


@dataclass(frozen=True)
class PromptGroupLease:
    lease_id: str
    group_id: str
    behavior_version: PolicyVersion
    payload: Any
    attempt_id: int = 0


@dataclass(frozen=True)
class RolloutOutput:
    """Generated tokens and their behavior-policy log probabilities.

    Log probabilities must describe the distribution that actually sampled
    each token.  In particular, tt-transformers currently samples with a
    top-k filter in some configurations.  Whether the resulting truncated
    behavior policy has a material effect on importance sampling still needs
    to be measured; callers must not silently substitute full-policy scores.
    """

    tokens: tuple[tuple[int, ...], ...]
    logprobs: tuple[tuple[float, ...], ...]

    @classmethod
    def from_sequences(
        cls,
        tokens: Sequence[Sequence[int]],
        logprobs: Sequence[Sequence[float]],
    ) -> "RolloutOutput":
        return cls(
            tokens=tuple(tuple(int(token) for token in row) for row in tokens),
            logprobs=tuple(tuple(float(logprob) for logprob in row) for row in logprobs),
        )

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.logprobs):
            raise ValueError("tokens and logprobs must have the same batch size")
        for row, (tokens, logprobs) in enumerate(zip(self.tokens, self.logprobs)):
            if len(tokens) != len(logprobs):
                raise ValueError(
                    f"tokens and logprobs must have the same length for row {row} "
                    f"(got {len(tokens)} and {len(logprobs)})"
                )


@dataclass(frozen=True)
class RolloutResult:
    engine_id: str
    lease_id: str
    group_id: str
    attempt_id: int
    behavior_version: PolicyVersion
    output: RolloutOutput


@dataclass(frozen=True)
class ResultReady:
    result: RolloutResult


@dataclass(frozen=True)
class WeightsStaged:
    engine_id: str
    version: PolicyVersion


@dataclass(frozen=True)
class PolicyActivated:
    engine_id: str
    version: PolicyVersion


@dataclass(frozen=True)
class EngineFailed:
    engine_id: str
    operation: str
    message: str
    active_version: PolicyVersion
    target_version: Optional[PolicyVersion]
    lease_id: Optional[str]


EngineEvent = ResultReady | WeightsStaged | PolicyActivated | EngineFailed


@dataclass(frozen=True)
class EngineSnapshot:
    engine_id: str
    state: EngineState
    active_version: PolicyVersion
    target_version: Optional[PolicyVersion]
    staging_version: Optional[PolicyVersion]
    staged_version: Optional[PolicyVersion]
    active_lease_id: Optional[str]
    failure: Optional[str]


class RolloutEngine(ABC):
    """Template-method base class for one independently schedulable engine.

    Public lifecycle methods are final so every backend observes the same
    transition rules.  A re-entrant lock protects short state mutations, but no
    backend action executes while that lock is held.
    """

    MAX_INFLIGHT_PROMPT_GROUPS: Final[int] = 1

    def __init__(
        self,
        *,
        engine_id: str,
        active_version: PolicyVersion,
        event_sink: Optional[EventSink] = None,
    ) -> None:
        if not engine_id:
            raise ValueError("engine_id must be non-empty")
        if active_version < 0:
            raise ValueError("active_version must be non-negative")

        self._engine_id = engine_id
        self._state = EngineState.READY
        self._active_version = active_version
        self._target_version: Optional[PolicyVersion] = None
        self._staging_version: Optional[PolicyVersion] = None
        self._staged_version: Optional[PolicyVersion] = None
        self._active_lease: Optional[PromptGroupLease] = None
        self._failure: Optional[str] = None
        self._event_sink: EventSink = event_sink or (lambda _event: None)
        self._lock = RLock()

    @property
    def engine_id(self) -> str:
        return self._engine_id

    @final
    def snapshot(self) -> EngineSnapshot:
        with self._lock:
            return EngineSnapshot(
                engine_id=self._engine_id,
                state=self._state,
                active_version=self._active_version,
                target_version=self._target_version,
                staging_version=self._staging_version,
                staged_version=self._staged_version,
                active_lease_id=self._active_lease.lease_id if self._active_lease else None,
                failure=self._failure,
            )

    @final
    def start_rollout(self, lease: PromptGroupLease) -> None:
        """Claim ``lease`` and initiate one atomic prompt-group rollout."""
        with self._lock:
            self._require_state(EngineState.READY, "start rollout")
            if not lease.lease_id or not lease.group_id:
                raise ValueError("lease_id and group_id must be non-empty")
            if lease.attempt_id < 0:
                raise ValueError("attempt_id must be non-negative")
            if lease.behavior_version != self._active_version:
                raise InvalidTransitionError(
                    f"lease behavior version {lease.behavior_version} does not match "
                    f"engine active version {self._active_version}"
                )
            self._active_lease = lease
            self._state = EngineState.GENERATING

        self._invoke_action("start_rollout", self._start_rollout_action, lease)

    @final
    def quiesce(self, target_version: PolicyVersion) -> None:
        """Refuse new leases and drain toward ``target_version``.

        ``stage_weights`` also establishes the drain target.  Supporting both
        arrival orders keeps separate control and weight-transfer lanes safe.
        Repeating the same quiesce command is idempotent.
        """
        activation = None
        with self._lock:
            self._require_operational("quiesce")
            self._establish_target(target_version)
            self._enter_draining_state()
            activation = self._prepare_activation_if_ready()

        if activation is not None:
            self._invoke_activation_action(activation)

    @final
    def stage_weights(self, version: PolicyVersion, source: Any) -> None:
        """Initiate receipt of ``version`` into inactive backend-owned storage.

        Receipt of a new-version stage request is itself enough to stop new
        leases.  An explicitly delivered ``quiesce(version)`` is therefore
        idempotent and message ordering cannot start extra old-version work.
        """
        with self._lock:
            self._require_operational("stage weights")
            self._establish_target(version)
            if self._staging_version is not None or self._staged_version is not None:
                existing = self._staging_version if self._staging_version is not None else self._staged_version
                raise InvalidTransitionError(
                    f"cannot stage version {version}; version {existing} already occupies staging"
                )
            self._staging_version = version
            self._enter_draining_state()

        self._invoke_stage_action(version, source)

    @final
    def rollout_completed(self, lease_id: str, output: RolloutOutput) -> None:
        """Report successful completion from ``_start_rollout_action``."""
        activation = None
        with self._lock:
            if self._state not in (EngineState.GENERATING, EngineState.DRAINING):
                raise InvalidTransitionError(f"rollout completion is invalid while state is {self._state.name}")
            lease = self._active_lease
            if lease is None or lease.lease_id != lease_id:
                actual = lease.lease_id if lease else None
                raise InvalidTransitionError(f"completion lease {lease_id!r} does not match active lease {actual!r}")

            result = RolloutResult(
                engine_id=self._engine_id,
                lease_id=lease.lease_id,
                group_id=lease.group_id,
                attempt_id=lease.attempt_id,
                behavior_version=lease.behavior_version,
                output=output,
            )
            self._active_lease = None
            if self._target_version is None:
                self._state = EngineState.READY
            else:
                self._state = EngineState.WAITING_FOR_STAGE
                activation = self._prepare_activation_if_ready()

        self._emit(ResultReady(result))
        if activation is not None:
            self._invoke_activation_action(activation)

    @final
    def rollout_failed(self, lease_id: str, error: BaseException | str) -> None:
        with self._lock:
            lease = self._active_lease
            if lease is None or lease.lease_id != lease_id:
                actual = lease.lease_id if lease else None
                raise InvalidTransitionError(f"failed lease {lease_id!r} does not match active lease {actual!r}")
        self._fail("rollout", error)

    @final
    def weights_staged(self, version: PolicyVersion) -> None:
        """Report a complete, validated staging transfer."""
        activation = None
        with self._lock:
            self._require_operational("complete weight staging")
            if version != self._staging_version:
                raise InvalidTransitionError(
                    f"staged completion for version {version} does not match "
                    f"in-progress version {self._staging_version}"
                )
            self._staging_version = None
            self._staged_version = version
            activation = self._prepare_activation_if_ready()

        self._emit(WeightsStaged(self._engine_id, version))
        if activation is not None:
            self._invoke_activation_action(activation)

    @final
    def weight_staging_failed(self, version: PolicyVersion, error: BaseException | str) -> None:
        with self._lock:
            if version != self._staging_version:
                raise InvalidTransitionError(
                    f"staging failure for version {version} does not match in-progress version {self._staging_version}"
                )
            self._staging_version = None
        self._discard_and_fail("stage_weights", version, error)

    @final
    def activation_completed(self, version: PolicyVersion) -> None:
        """Report that staged weights have been copied into active buffers."""
        with self._lock:
            self._require_state(EngineState.ACTIVATING, "complete activation")
            if version != self._target_version or version != self._staged_version:
                raise InvalidTransitionError(
                    f"activation completion {version} does not match target/staged versions "
                    f"{self._target_version}/{self._staged_version}"
                )
            self._active_version = version
            self._target_version = None
            self._staged_version = None
            self._state = EngineState.READY

        self._emit(PolicyActivated(self._engine_id, version))

    @final
    def activation_failed(self, version: PolicyVersion, error: BaseException | str) -> None:
        with self._lock:
            self._require_state(EngineState.ACTIVATING, "fail activation")
            if version != self._target_version:
                raise InvalidTransitionError(
                    f"activation failure {version} does not match target version {self._target_version}"
                )
        self._discard_and_fail("activate", version, error)

    @final
    def shutdown(self) -> None:
        """Stop an idle engine. Active work must first complete or fail."""
        with self._lock:
            self._require_state(EngineState.READY, "shut down")
            self._state = EngineState.STOPPED
        self._invoke_action("shutdown", self._shutdown_action)

    def _establish_target(self, version: PolicyVersion) -> None:
        if version <= self._active_version:
            raise InvalidTransitionError(
                f"target version {version} must be newer than active version {self._active_version}"
            )
        if self._target_version is not None and version != self._target_version:
            raise InvalidTransitionError(
                f"target version {version} conflicts with in-progress target {self._target_version}"
            )
        self._target_version = version

    def _enter_draining_state(self) -> None:
        # A repeated quiesce may arrive after staging already triggered
        # activation. It acknowledges the same target but must not rewind the
        # state or launch a second activation action.
        if self._state is EngineState.ACTIVATING:
            return
        self._state = EngineState.DRAINING if self._active_lease is not None else EngineState.WAITING_FOR_STAGE

    def _prepare_activation_if_ready(self) -> Optional[PolicyVersion]:
        if (
            self._target_version is not None
            and self._active_lease is None
            and self._staged_version == self._target_version
            and self._state is not EngineState.ACTIVATING
        ):
            self._state = EngineState.ACTIVATING
            return self._target_version
        return None

    def _require_operational(self, operation: str) -> None:
        if self._state in (EngineState.FAILED, EngineState.STOPPED):
            raise InvalidTransitionError(f"cannot {operation} while state is {self._state.name}")

    def _require_state(self, expected: EngineState, operation: str) -> None:
        if self._state is not expected:
            raise InvalidTransitionError(
                f"cannot {operation} while state is {self._state.name}; expected {expected.name}"
            )

    def _invoke_stage_action(self, version: PolicyVersion, source: Any) -> None:
        try:
            self._stage_weights_action(version, source)
        except Exception as error:
            with self._lock:
                if self._staging_version == version:
                    self._staging_version = None
            self._discard_and_fail("stage_weights", version, error)
            raise

    def _invoke_activation_action(self, version: PolicyVersion) -> None:
        try:
            self._activate_staged_action(version)
        except Exception as error:
            self._discard_and_fail("activate", version, error)
            raise

    def _discard_and_fail(self, operation: str, version: PolicyVersion, error: BaseException | str) -> None:
        failure: BaseException | str = error
        try:
            self._discard_staged_action(version)
        except Exception as discard_error:
            failure = f"{error}; additionally failed to discard staged version {version}: {discard_error}"
        else:
            with self._lock:
                if self._staging_version == version:
                    self._staging_version = None
                if self._staged_version == version:
                    self._staged_version = None
        self._fail(operation, failure)

    def _invoke_action(self, operation: str, action: Callable[..., None], *args: Any) -> None:
        try:
            action(*args)
        except Exception as error:
            self._fail(operation, error)
            raise

    def _fail(self, operation: str, error: BaseException | str) -> None:
        message = str(error)
        with self._lock:
            if self._state is EngineState.FAILED:
                return
            lease_id = self._active_lease.lease_id if self._active_lease else None
            self._failure = f"{operation}: {message}"
            self._state = EngineState.FAILED
            event = EngineFailed(
                engine_id=self._engine_id,
                operation=operation,
                message=message,
                active_version=self._active_version,
                target_version=self._target_version,
                lease_id=lease_id,
            )
        self._emit(event)

    def _emit(self, event: EngineEvent) -> None:
        self._event_sink(event)

    @abstractmethod
    def _start_rollout_action(self, lease: PromptGroupLease) -> None:
        """Initiate backend generation, then call a rollout completion method."""

    @abstractmethod
    def _stage_weights_action(self, version: PolicyVersion, source: Any) -> None:
        """Initiate receive/validation into inactive storage."""

    @abstractmethod
    def _activate_staged_action(self, version: PolicyVersion) -> None:
        """Initiate the safe-boundary device-local activation."""

    @abstractmethod
    def _discard_staged_action(self, version: PolicyVersion) -> None:
        """Discard or invalidate an incomplete/failed staged version."""

    def _shutdown_action(self) -> None:
        """Optional backend cleanup after the base class enters STOPPED."""


__all__ = [
    "EngineEvent",
    "EngineFailed",
    "EngineSnapshot",
    "EngineState",
    "InvalidTransitionError",
    "PolicyActivated",
    "PromptGroupLease",
    "ResultReady",
    "RolloutEngine",
    "RolloutResult",
    "RolloutOutput",
    "WeightsStaged",
]

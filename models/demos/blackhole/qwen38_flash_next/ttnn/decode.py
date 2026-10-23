# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Strict ordinary greedy-decode session for Qwen3.8-Flash-Next TTNN.

This module is the request/state owner above :class:`Qwen38TTNNTextModel`.
It deliberately does not open or close a device, acquire a lease, convert a
weight, implement sampling, or render chat.  A launcher must first open one
qualified physical ``1x4`` Blackhole mesh under its full-lifetime locks and
enforcing broker lease, construct :class:`Qwen38TTNNBuilder`, and supply the
independently recorded live identity and provenance here.

The correctness baseline is fully serialized.  Each prompt token and each
generated-token continuation is one true-global-B1 target invocation bracketed
by explicit mesh synchronization.  The reported ``model_call_ns`` is host wall
time for the Python/TTNN call and can include implicit synchronization needed
to return the CPU greedy candidate.  It is not a device-profiler measurement.
``pre_sync_ns`` and ``post_sync_ns`` are recorded separately, and
``end_to_end_ns`` spans the entire token or TTFT boundary.  These domains must
not be added to device-profiler or Tracy timings.

The state position counts tokens already consumed by the target.  A generated
token is emitted before it is consumed, so ``pending_token_id`` is explicit.
Continuing a completed request without resetting requires the next prompt
suffix to begin with that pending token.  This prevents a later chat owner from
silently appending new text to a cache which does not contain the previous
assistant terminator.

``stop_on_eos=False`` exists only for bounded benchmark runs that must collect
a fixed sample count even when an early greedy token is in the pinned EOS set.
It never disables the independent ``max_new_tokens`` or allocated-context
bounds.  Production callers retain the default EOS termination behavior.

Any model-call, state-validation, cleanup, clock, or synchronization failure
poisons this session permanently.  If the transaction-hardened model reports
itself poisoned, its error is retained as the cause.  A poisoned session never
calls reset or release methods: the enclosing task-owned hardware process must
terminate and release its already-held lease without resetting hardware.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, NoReturn

import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import (
    Qwen38BuildProvenance,
    Qwen38LiveBuildIdentity,
    Qwen38TTNNBuilder,
    Qwen38TTNNBuiltTarget,
    Qwen38TTNNTargetComponents,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE, TP_SIZE, Qwen38MeshContract
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import BACKBONE_LAYERS
from models.demos.blackhole.qwen38_flash_next.ttnn.model import (
    MAX_CONTEXT,
    VOCAB_SIZE,
    DecodePhaseObserver,
    LayerObserver,
    Qwen38TTNNGreedyStep,
    Qwen38TTNNTextModel,
    Qwen38TTNNTextModelOutput,
    Qwen38TTNNTextModelState,
)

IdentityKey = str
Clock = Callable[[], int]
Synchronize = Callable[[], None]


class Qwen38OrdinarySessionStatus(str, Enum):
    """Lifecycle of one model-state owner."""

    IDLE = "idle"
    ACTIVE = "active"
    FINISHED = "finished"
    POISONED = "poisoned"
    CLOSED = "closed"


class Qwen38OrdinaryStopReason(str, Enum):
    EOS = "eos"
    MAX_NEW_TOKENS = "max_new_tokens"
    CONTEXT_LENGTH = "context_length"


class Qwen38OrdinaryTimingPhase(str, Enum):
    TTFT = "ttft"
    DECODE = "decode"


class Qwen38OrdinaryDecodeError(RuntimeError):
    """Base exception for the strict ordinary-decode owner."""


class Qwen38OrdinaryDecodePoisonedError(Qwen38OrdinaryDecodeError):
    """The model/cache state is no longer safe to inspect or reuse."""

    def __init__(self, operation: str, cause: BaseException) -> None:
        self.operation = operation
        self.cause = cause
        super().__init__(
            f"ordinary TTNN decode was poisoned during {operation}: {type(cause).__name__}: {cause}; "
            "terminate the task-owned process and release its lease without resetting hardware"
        )


@dataclass(frozen=True)
class Qwen38OrdinaryInvocationTiming:
    """Synchronized wall-clock domains for one consumed target token."""

    input_token_id: int
    input_position: int
    pre_sync_ns: int
    model_call_ns: int
    post_sync_ns: int
    end_to_end_ns: int

    @property
    def explicit_sync_ns(self) -> int:
        return self.pre_sync_ns + self.post_sync_ns

    @property
    def host_overhead_ns(self) -> int:
        return self.end_to_end_ns - self.model_call_ns - self.explicit_sync_ns


@dataclass(frozen=True)
class Qwen38OrdinaryEmissionTiming:
    """TTFT or one steady-decode boundary with its serial invocations."""

    phase: Qwen38OrdinaryTimingPhase
    input_token_count: int
    model_call_ns: int
    explicit_sync_ns: int
    host_overhead_ns: int
    end_to_end_ns: int
    invocations: tuple[Qwen38OrdinaryInvocationTiming, ...]


@dataclass(frozen=True)
class Qwen38OrdinaryToken:
    """One CPU greedy token emitted by the exact ordinary target."""

    request_id: int
    token_index: int
    token_id: int
    cache_position: int
    stop_reason: Qwen38OrdinaryStopReason | None
    timing: Qwen38OrdinaryEmissionTiming


@dataclass(frozen=True)
class Qwen38OrdinarySessionIdentity:
    """Inspectable launch identity retained by the request owner."""

    identity_key: IdentityKey
    provenance: Qwen38BuildProvenance
    physical_ids: tuple[int, int, int, int]
    mesh_shape: tuple[int, int]
    collective_topology: str
    dram_bank_ring_order: tuple[int, ...]
    ring_size: int


def _require_identity_key(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"expected_identity_key must be lowercase 64-hex, got {value!r}")
    return value


def _normalize_physical_ids(values: Sequence[int]) -> tuple[int, int, int, int]:
    result = tuple(values)
    if (
        len(result) != TP_SIZE
        or len(set(result)) != TP_SIZE
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in result)
    ):
        raise ValueError(f"ordinary decode requires four distinct nonnegative physical IDs, got {result}")
    return result  # type: ignore[return-value]


def _normalize_eos_ids(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(values)
    if not result:
        raise ValueError("at least one pinned EOS token ID is required")
    if len(set(result)) != len(result):
        raise ValueError(f"EOS token IDs must be unique, got {result}")
    if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < VOCAB_SIZE for value in result):
        raise ValueError(f"EOS token IDs must be integers in [0,{VOCAB_SIZE}), got {result}")
    return result


def _normalize_prompt(input_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
    if isinstance(input_ids, torch.Tensor):
        if input_ids.device.type != "cpu" or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("ordinary prompt IDs must be CPU int32/int64")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] <= 0:
            raise ValueError("ordinary prompt IDs must be nonempty true-global-B1 [1,sequence]")
        result = input_ids.to(dtype=torch.long).contiguous()
    else:
        if isinstance(input_ids, (str, bytes)) or not isinstance(input_ids, Sequence):
            raise TypeError("ordinary prompt IDs must be a CPU tensor or a flat integer sequence")
        values = tuple(input_ids)
        if not values:
            raise ValueError("ordinary prompt IDs cannot be empty")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("ordinary prompt IDs must be a flat sequence of integers")
        result = torch.tensor([values], dtype=torch.long)
    if int(result.min()) < 0 or int(result.max()) >= VOCAB_SIZE:
        raise IndexError(f"ordinary prompt token is outside [0,{VOCAB_SIZE})")
    return result


def _normalize_max_new_tokens(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"max_new_tokens must be a positive integer, got {value!r}")
    return value


def _duration(start: int, end: int, *, label: str) -> int:
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
        raise RuntimeError(f"{label} clock must return integer nanoseconds")
    if end < start:
        raise RuntimeError(f"{label} monotonic clock moved backwards: {start} -> {end}")
    return end - start


def _validate_target(
    built_target: Qwen38TTNNBuiltTarget,
    *,
    expected_provenance: Qwen38BuildProvenance,
    expected_physical_ids: tuple[int, int, int, int],
    expected_identity_key: str,
) -> tuple[Qwen38TTNNTextModel, Qwen38LiveBuildIdentity, int]:
    if type(built_target) is not Qwen38TTNNBuiltTarget:
        raise TypeError("ordinary decode requires the exact Qwen38TTNNBuiltTarget owner")
    if type(built_target.model) is not Qwen38TTNNTextModel:
        raise TypeError("ordinary decode requires the exact Qwen38TTNNTextModel")
    if type(built_target.components) is not Qwen38TTNNTargetComponents:
        raise TypeError("ordinary decode requires the exact Qwen38TTNNTargetComponents owner")
    if type(expected_provenance) is not Qwen38BuildProvenance:
        raise TypeError("expected_provenance must be a validated Qwen38BuildProvenance")

    model = built_target.model
    components = built_target.components
    identity = components.identity
    if type(identity) is not Qwen38LiveBuildIdentity:
        raise TypeError("target components do not carry the exact live builder identity")
    if identity.provenance != expected_provenance:
        raise ValueError("built target provenance differs from the independently supplied launch provenance")
    if identity.key != expected_identity_key:
        raise ValueError(f"built target identity {identity.key} differs from expected {expected_identity_key}")
    if identity.physical_ids != expected_physical_ids:
        raise ValueError(
            f"built target physical order {identity.physical_ids} differs from expected {expected_physical_ids}"
        )
    if tuple(identity.mesh_shape) != MESH_SHAPE:
        raise ValueError(f"ordinary decode requires logical mesh {MESH_SHAPE}, got {identity.mesh_shape}")
    if type(model.mesh_contract) is not Qwen38MeshContract:
        raise TypeError("ordinary target does not carry the exact Qwen38MeshContract")
    if model.mesh_contract.physical_ids != expected_physical_ids:
        raise ValueError("model mesh contract differs from the launch physical-device order")
    model.mesh_contract.validate_mesh(model.mesh_device)
    if model.model_io is not components.model_io:
        raise ValueError("model I/O is not owned by the supplied target components")
    if model.layers is not components.layers:
        raise ValueError("model layer tuple is not owned by the supplied target components")
    if model.final_mixer is not components.final_mixer:
        raise ValueError("model final mixer is not owned by the supplied target components")
    if len(model.layers) != BACKBONE_LAYERS:
        raise ValueError(f"ordinary target must contain exactly {BACKBONE_LAYERS} layers")
    if any(layer.expert_streamer is not components.expert_streamer for layer in model.layers):
        raise ValueError("ordinary target layers do not share the component owner's single BF4_B streamer")
    allocated_context = getattr(model, "allocated_context", None)
    if (
        isinstance(allocated_context, bool)
        or not isinstance(allocated_context, int)
        or not 0 < allocated_context <= MAX_CONTEXT
    ):
        raise ValueError(f"ordinary target allocated context is invalid: {allocated_context!r}")
    if identity.qsa_cache_capacity != allocated_context:
        raise ValueError(
            "ordinary target allocated context differs from its authenticated live build identity: "
            f"{allocated_context} != {identity.qsa_cache_capacity}"
        )
    return model, identity, allocated_context


class Qwen38OrdinaryDecodeSession:
    """Single-sequence, exact-greedy owner for one already-open 1x4 mesh."""

    def __init__(
        self,
        built_target: Qwen38TTNNBuiltTarget,
        *,
        expected_provenance: Qwen38BuildProvenance,
        expected_physical_ids: Sequence[int],
        expected_identity_key: str,
        eos_token_ids: Sequence[int],
        stop_on_eos: bool = True,
        layer_observer: LayerObserver | None = None,
        phase_observer: DecodePhaseObserver | None = None,
        synchronize: Synchronize | None = None,
        clock_ns: Clock = time.perf_counter_ns,
    ) -> None:
        physical_ids = _normalize_physical_ids(expected_physical_ids)
        identity_key = _require_identity_key(expected_identity_key)
        eos_ids = _normalize_eos_ids(eos_token_ids)
        if type(stop_on_eos) is not bool:
            raise TypeError("stop_on_eos must be an explicit bool")
        if synchronize is not None and not callable(synchronize):
            raise TypeError("synchronize must be callable")
        if layer_observer is not None and not callable(layer_observer):
            raise TypeError("layer_observer must be callable")
        if phase_observer is not None and not callable(phase_observer):
            raise TypeError("phase_observer must be callable")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        model, identity, allocated_context = _validate_target(
            built_target,
            expected_provenance=expected_provenance,
            expected_physical_ids=physical_ids,
            expected_identity_key=identity_key,
        )

        self._built_target = built_target
        self.model = model
        self.allocated_context = allocated_context
        self.identity = Qwen38OrdinarySessionIdentity(
            identity_key=identity.key,
            provenance=identity.provenance,
            physical_ids=identity.physical_ids,
            mesh_shape=identity.mesh_shape,
            collective_topology=identity.collective_topology,
            dram_bank_ring_order=identity.dram_bank_ring_order,
            ring_size=identity.ring_size,
        )
        self.eos_token_ids = eos_ids
        self.stop_on_eos = stop_on_eos
        self._eos_token_set = frozenset(eos_ids) if stop_on_eos else frozenset()
        self._synchronize = synchronize or (lambda: ttnn.synchronize_device(self.model.mesh_device))
        self._layer_observer = layer_observer
        self._phase_observer = phase_observer
        self._clock_ns = clock_ns
        self._status = Qwen38OrdinarySessionStatus.IDLE
        self._poison_error: Qwen38OrdinaryDecodePoisonedError | None = None
        self._state: Qwen38TTNNTextModelState | None = None
        self._pending_token_id: int | None = None
        self._request_id = 0
        self._request_max_new_tokens = 0
        self._request_generated = 0
        self._metrics: list[Qwen38OrdinaryToken] = []
        self._runtime_owner = object()

        self.model.claim_runtime_owner(self._runtime_owner)

        try:
            state = self.model.allocate_state()
            self._validate_state_position(state, expected=0)
            self._state = state
            self._timed_sync()
            self._raise_if_model_poisoned("allocate_state")
        except BaseException as error:
            self._poison("allocate_state", error)

    @classmethod
    def from_builder(
        cls,
        builder: Qwen38TTNNBuilder,
        *,
        expected_provenance: Qwen38BuildProvenance,
        expected_physical_ids: Sequence[int],
        expected_identity_key: str,
        eos_token_ids: Sequence[int],
        stop_on_eos: bool = True,
        layer_observer: LayerObserver | None = None,
        phase_observer: DecodePhaseObserver | None = None,
        synchronize: Synchronize | None = None,
        clock_ns: Clock = time.perf_counter_ns,
    ) -> Qwen38OrdinaryDecodeSession:
        """Build the exact target without opening a mesh, then own its state."""

        if type(builder) is not Qwen38TTNNBuilder:
            raise TypeError("from_builder requires the exact Qwen38TTNNBuilder")
        physical_ids = _normalize_physical_ids(expected_physical_ids)
        identity_key = _require_identity_key(expected_identity_key)
        eos_ids = _normalize_eos_ids(eos_token_ids)
        if type(stop_on_eos) is not bool:
            raise TypeError("stop_on_eos must be an explicit bool")
        if synchronize is not None and not callable(synchronize):
            raise TypeError("synchronize must be callable")
        if layer_observer is not None and not callable(layer_observer):
            raise TypeError("layer_observer must be callable")
        if phase_observer is not None and not callable(phase_observer):
            raise TypeError("phase_observer must be callable")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        if builder.provenance != expected_provenance:
            raise ValueError("builder provenance differs from independently supplied launch provenance")
        if builder.identity.physical_ids != physical_ids or builder.identity.key != identity_key:
            raise ValueError("builder live identity differs from the admitted launch identity")
        return cls(
            builder.build_target(),
            expected_provenance=expected_provenance,
            expected_physical_ids=physical_ids,
            expected_identity_key=identity_key,
            eos_token_ids=eos_ids,
            stop_on_eos=stop_on_eos,
            layer_observer=layer_observer,
            phase_observer=phase_observer,
            synchronize=synchronize,
            clock_ns=clock_ns,
        )

    @property
    def status(self) -> Qwen38OrdinarySessionStatus:
        return self._status

    @property
    def state_position(self) -> int:
        self._assert_usable("state_position")
        return self._require_state().position

    @property
    def pending_token_id(self) -> int | None:
        return self._pending_token_id

    @property
    def metrics(self) -> tuple[Qwen38OrdinaryToken, ...]:
        return tuple(self._metrics)

    @property
    def poison_error(self) -> Qwen38OrdinaryDecodePoisonedError | None:
        return self._poison_error

    def _model_poison_error(self) -> BaseException | None:
        try:
            poisoned = bool(getattr(self.model, "poisoned", False))
        except BaseException as error:
            return error
        if not poisoned:
            return None
        try:
            model_error = getattr(self.model, "poisoned_error", None)
        except BaseException as error:
            return error
        if isinstance(model_error, BaseException):
            return model_error
        return RuntimeError("the TTNN text-model owner reports poison without an exception")

    def _raise_if_model_poisoned(self, operation: str) -> None:
        model_error = self._model_poison_error()
        if model_error is not None:
            self._poison(operation, model_error)

    def _poison(self, operation: str, error: BaseException) -> NoReturn:
        if self._poison_error is None:
            model_error = self._model_poison_error()
            cause = model_error if model_error is not None else error
            self._poison_error = Qwen38OrdinaryDecodePoisonedError(operation, cause)
        self._status = Qwen38OrdinarySessionStatus.POISONED
        raise self._poison_error from error

    def _assert_usable(self, operation: str) -> None:
        if self._status is Qwen38OrdinarySessionStatus.POISONED:
            assert self._poison_error is not None
            raise self._poison_error
        if self._status is Qwen38OrdinarySessionStatus.CLOSED:
            raise Qwen38OrdinaryDecodeError(f"cannot {operation}: ordinary decode session is closed")
        self._raise_if_model_poisoned(operation)

    def _require_state(self) -> Qwen38TTNNTextModelState:
        if self._state is None:
            raise RuntimeError("ordinary decode state is not allocated")
        return self._state

    def _validate_state_position(self, state: Any, *, expected: int) -> None:
        if type(state) is not Qwen38TTNNTextModelState:
            raise TypeError("ordinary target returned a non-Qwen38 TTNN model state")
        if state._owner is not self.model._state_owner:
            raise ValueError("ordinary target returned state from a different text-model owner")
        if state.position != expected:
            raise RuntimeError(f"ordinary target state advanced to {state.position}, expected {expected}")

    def _timed_sync(self) -> int:
        started = self._clock_ns()
        self._synchronize()
        ended = self._clock_ns()
        return _duration(started, ended, label="synchronize")

    def _invoke(self, token_id: int, *, greedy: bool) -> tuple[int | None, Qwen38OrdinaryInvocationTiming]:
        try:
            state = self._require_state()
            position = state.position
            invocation_started = self._clock_ns()
            pre_sync_ns = self._timed_sync()
            call_started = self._clock_ns()
            if greedy:
                if self._phase_observer is None:
                    step = self.model.greedy_step(
                        token_id,
                        state,
                        retain_input_state=False,
                        layer_observer=self._layer_observer,
                    )
                else:
                    step = self.model.greedy_step(
                        token_id,
                        state,
                        retain_input_state=False,
                        layer_observer=self._layer_observer,
                        phase_observer=self._phase_observer,
                    )
                call_ended = self._clock_ns()
                if type(step) is not Qwen38TTNNGreedyStep:
                    raise TypeError("ordinary target greedy step returned the wrong owner type")
                if step.input_token_id != token_id:
                    raise RuntimeError(
                        f"ordinary target reports input token {step.input_token_id}, expected {token_id}"
                    )
                self._validate_state_position(step.state, expected=position + 1)
                if len(step.layer_aux) != BACKBONE_LAYERS:
                    raise RuntimeError(f"ordinary target returned {len(step.layer_aux)} layer records, expected 48")
                if isinstance(step.next_token_id, bool) or not 0 <= step.next_token_id < VOCAB_SIZE:
                    raise RuntimeError(f"ordinary target returned invalid greedy token {step.next_token_id!r}")
                self._state = step.state
                next_token: int | None = step.next_token_id
            else:
                if self._phase_observer is None:
                    output = self.model.forward_decode(
                        token_id,
                        state,
                        return_logits=False,
                        resolve_greedy=False,
                        retain_hidden=False,
                        retain_hyper_residual=False,
                        retain_input_state=False,
                        return_routing=False,
                        layer_observer=self._layer_observer,
                    )
                else:
                    output = self.model.forward_decode(
                        token_id,
                        state,
                        return_logits=False,
                        resolve_greedy=False,
                        retain_hidden=False,
                        retain_hyper_residual=False,
                        retain_input_state=False,
                        return_routing=False,
                        layer_observer=self._layer_observer,
                        phase_observer=self._phase_observer,
                    )
                call_ended = self._clock_ns()
                if type(output) is not Qwen38TTNNTextModelOutput:
                    raise TypeError("ordinary target forward returned the wrong owner type")
                if output.input_token_id != token_id or output.position != position:
                    raise RuntimeError("ordinary target forward returned mismatched token/position metadata")
                self._validate_state_position(output.state, expected=position + 1)
                if len(output.layer_aux) != BACKBONE_LAYERS:
                    raise RuntimeError(f"ordinary target returned {len(output.layer_aux)} layer records, expected 48")
                if any(
                    value is not None
                    for value in (
                        output.hyper_residual_sharded,
                        output.hidden_sharded,
                        output.logits,
                        output.greedy_token,
                    )
                ):
                    raise RuntimeError("non-final serial prompt step retained an unexpected output tensor")
                self._state = output.state
                output.release_tensors()
                next_token = None
            model_call_ns = _duration(call_started, call_ended, label="model call")
            post_sync_ns = self._timed_sync()
            self._raise_if_model_poisoned("greedy_step" if greedy else "forward_decode")
            invocation_ended = self._clock_ns()
            end_to_end_ns = _duration(invocation_started, invocation_ended, label="target invocation")
            timing = Qwen38OrdinaryInvocationTiming(
                input_token_id=token_id,
                input_position=position,
                pre_sync_ns=pre_sync_ns,
                model_call_ns=model_call_ns,
                post_sync_ns=post_sync_ns,
                end_to_end_ns=end_to_end_ns,
            )
            if timing.host_overhead_ns < 0:
                raise RuntimeError("ordinary invocation timing domains overlap")
            return next_token, timing
        except Qwen38OrdinaryDecodePoisonedError:
            raise
        except BaseException as error:
            self._poison("greedy_step" if greedy else "forward_decode", error)

    @staticmethod
    def _emission_timing(
        phase: Qwen38OrdinaryTimingPhase,
        started: int,
        ended: int,
        invocations: Sequence[Qwen38OrdinaryInvocationTiming],
    ) -> Qwen38OrdinaryEmissionTiming:
        invocation_tuple = tuple(invocations)
        if not invocation_tuple:
            raise RuntimeError("an emission timing requires at least one target invocation")
        end_to_end_ns = _duration(started, ended, label=phase.value)
        model_call_ns = sum(item.model_call_ns for item in invocation_tuple)
        explicit_sync_ns = sum(item.explicit_sync_ns for item in invocation_tuple)
        host_overhead_ns = end_to_end_ns - model_call_ns - explicit_sync_ns
        if host_overhead_ns < 0:
            raise RuntimeError(f"{phase.value} timing domains overlap")
        return Qwen38OrdinaryEmissionTiming(
            phase=phase,
            input_token_count=len(invocation_tuple),
            model_call_ns=model_call_ns,
            explicit_sync_ns=explicit_sync_ns,
            host_overhead_ns=host_overhead_ns,
            end_to_end_ns=end_to_end_ns,
            invocations=invocation_tuple,
        )

    def _stop_reason(self, token_id: int) -> Qwen38OrdinaryStopReason | None:
        if token_id in self._eos_token_set:
            return Qwen38OrdinaryStopReason.EOS
        if self._request_generated >= self._request_max_new_tokens:
            return Qwen38OrdinaryStopReason.MAX_NEW_TOKENS
        if self._require_state().position >= self.allocated_context:
            return Qwen38OrdinaryStopReason.CONTEXT_LENGTH
        return None

    def _record_emission(
        self,
        token_id: int,
        *,
        timing: Qwen38OrdinaryEmissionTiming,
    ) -> Qwen38OrdinaryToken:
        self._request_generated += 1
        self._pending_token_id = token_id
        stop_reason = self._stop_reason(token_id)
        self._status = (
            Qwen38OrdinarySessionStatus.FINISHED if stop_reason is not None else Qwen38OrdinarySessionStatus.ACTIVE
        )
        event = Qwen38OrdinaryToken(
            request_id=self._request_id,
            token_index=self._request_generated - 1,
            token_id=token_id,
            cache_position=self._require_state().position,
            stop_reason=stop_reason,
            timing=timing,
        )
        self._metrics.append(event)
        return event

    def begin(
        self,
        input_ids: torch.Tensor | Sequence[int],
        *,
        max_new_tokens: int,
    ) -> Qwen38OrdinaryToken:
        """Consume an exact prompt suffix and return its first greedy token.

        On a nonzero cache, the suffix must begin with ``pending_token_id``.
        The first timing is TTFT and includes every serialized prompt-token
        invocation.  Input/template/tokenizer time before this call is outside
        the timing boundary.
        """

        self._assert_usable("begin a request")
        if self._status is Qwen38OrdinarySessionStatus.ACTIVE:
            raise Qwen38OrdinaryDecodeError("cannot begin a new request while decode is active")
        prompt = _normalize_prompt(input_ids)
        limit = _normalize_max_new_tokens(max_new_tokens)
        try:
            state = self._require_state()
        except BaseException as error:
            self._poison("begin preflight", error)
        prompt_length = int(prompt.shape[1])
        if state.position + prompt_length > self.allocated_context:
            raise ValueError(
                "ordinary prompt suffix exceeds the built target's allocated context " f"{self.allocated_context}"
            )
        if self._pending_token_id is not None and int(prompt[0, 0]) != self._pending_token_id:
            raise ValueError(
                f"continuation must first consume pending token {self._pending_token_id}, "
                f"got {int(prompt[0, 0])}; reset before replacing history"
            )

        try:
            request_started = self._clock_ns()
            self._request_id += 1
            self._request_generated = 0
            self._request_max_new_tokens = limit
            self._pending_token_id = None
            invocations: list[Qwen38OrdinaryInvocationTiming] = []
            next_token = None
            for offset in range(prompt_length):
                token_id = int(prompt[0, offset])
                next_token, invocation = self._invoke(token_id, greedy=offset == prompt_length - 1)
                invocations.append(invocation)
            if next_token is None:
                raise AssertionError("nonempty prompt did not produce a greedy token")
            request_ended = self._clock_ns()
            timing = self._emission_timing(
                Qwen38OrdinaryTimingPhase.TTFT,
                request_started,
                request_ended,
                invocations,
            )
            return self._record_emission(next_token, timing=timing)
        except Qwen38OrdinaryDecodePoisonedError:
            raise
        except BaseException as error:
            self._poison("begin", error)

    def step(self) -> Qwen38OrdinaryToken:
        """Consume the pending token and emit one steady ordinary-decode token."""

        self._assert_usable("decode a token")
        if self._status is not Qwen38OrdinarySessionStatus.ACTIVE:
            raise Qwen38OrdinaryDecodeError(f"cannot decode a token while session status is {self._status.value}")
        if self._pending_token_id is None:
            self._poison("step", RuntimeError("active ordinary decode has no pending token"))
        try:
            started = self._clock_ns()
            assert self._pending_token_id is not None
            next_token, invocation = self._invoke(self._pending_token_id, greedy=True)
            if next_token is None:
                self._poison("step", AssertionError("greedy continuation did not produce a token"))
            ended = self._clock_ns()
            timing = self._emission_timing(
                Qwen38OrdinaryTimingPhase.DECODE,
                started,
                ended,
                (invocation,),
            )
            assert next_token is not None
            return self._record_emission(next_token, timing=timing)
        except Qwen38OrdinaryDecodePoisonedError:
            raise
        except BaseException as error:
            self._poison("step", error)

    def generate(
        self,
        input_ids: torch.Tensor | Sequence[int],
        *,
        max_new_tokens: int,
    ) -> Iterator[Qwen38OrdinaryToken]:
        """Stream one request without adding sampling, chat, or MTP behavior."""

        event = self.begin(input_ids, max_new_tokens=max_new_tokens)
        yield event
        while self._status is Qwen38OrdinarySessionStatus.ACTIVE:
            yield self.step()

    def reset(self) -> None:
        """Reset a healthy state to position zero; never runs after poison."""

        self._assert_usable("reset")
        try:
            state = self.model.reset_state(self._require_state())
            self._validate_state_position(state, expected=0)
            self._state = state
            self._timed_sync()
            self._raise_if_model_poisoned("reset_state")
        except Qwen38OrdinaryDecodePoisonedError:
            raise
        except BaseException as error:
            self._poison("reset_state", error)
        self._pending_token_id = None
        self._request_max_new_tokens = 0
        self._request_generated = 0
        self._status = Qwen38OrdinarySessionStatus.IDLE

    def close(self) -> None:
        """Release only this session's healthy mutable state, exactly once."""

        if self._status is Qwen38OrdinarySessionStatus.CLOSED:
            return
        self._assert_usable("close")
        try:
            self.model.release_state(self._require_state())
            self._timed_sync()
            self._raise_if_model_poisoned("release_state")
        except Qwen38OrdinaryDecodePoisonedError:
            raise
        except BaseException as error:
            self._poison("release_state", error)
        try:
            self.model.release_runtime_owner(self._runtime_owner)
        except BaseException as error:
            self._poison("release_runtime_owner", error)
        self._state = None
        self._pending_token_id = None
        self._status = Qwen38OrdinarySessionStatus.CLOSED

    def __enter__(self) -> Qwen38OrdinaryDecodeSession:
        self._assert_usable("enter")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> Literal[False]:
        if self._status is not Qwen38OrdinarySessionStatus.POISONED:
            self.close()
        return False


def validate_decode_static_contract() -> None:
    """No-device guard for ordinary decode scope and termination semantics."""

    if (TP_SIZE, MESH_SHAPE, BACKBONE_LAYERS, VOCAB_SIZE, MAX_CONTEXT) != (
        4,
        (1, 4),
        48,
        248320,
        262144,
    ):
        raise RuntimeError("Qwen3.8 ordinary-decode geometry drifted")
    if tuple(item.value for item in Qwen38OrdinaryTimingPhase) != ("ttft", "decode"):
        raise RuntimeError("ordinary-decode timing domains drifted")
    if tuple(item.value for item in Qwen38OrdinaryStopReason) != (
        "eos",
        "max_new_tokens",
        "context_length",
    ):
        raise RuntimeError("ordinary-decode stop semantics drifted")


validate_decode_static_contract()


__all__ = [
    "Qwen38OrdinaryDecodeError",
    "Qwen38OrdinaryDecodePoisonedError",
    "Qwen38OrdinaryDecodeSession",
    "Qwen38OrdinaryEmissionTiming",
    "Qwen38OrdinaryInvocationTiming",
    "Qwen38OrdinarySessionIdentity",
    "Qwen38OrdinarySessionStatus",
    "Qwen38OrdinaryStopReason",
    "Qwen38OrdinaryTimingPhase",
    "Qwen38OrdinaryToken",
    "validate_decode_static_contract",
]

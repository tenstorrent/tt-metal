# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device lifecycle and fail-closed tests for ordinary TTNN decode."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.checkpoint import CHECKPOINT_FILE_MANIFEST_SHA256, INDEX_SHA256
from models.demos.blackhole.qwen38_flash_next.config import CONFIG_SHA256
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import (
    RESIDENT_MAX_QSA_CACHE_CAPACITY,
    Qwen38BuildProvenance,
    Qwen38LiveBuildIdentity,
    Qwen38TTNNBuilder,
    Qwen38TTNNBuiltTarget,
    Qwen38TTNNTargetComponents,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.decode import (
    Qwen38OrdinaryDecodeError,
    Qwen38OrdinaryDecodePoisonedError,
    Qwen38OrdinaryDecodeSession,
    Qwen38OrdinarySessionStatus,
    Qwen38OrdinaryStopReason,
    Qwen38OrdinaryTimingPhase,
    validate_decode_static_contract,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import (
    PINNED_CHECKPOINT_REVISION,
    PINNED_TENSOR_MANIFEST_SHA256,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import (
    BACKBONE_LAYERS,
    Qwen38TTNNDecoderLayerAux,
    Qwen38TTNNDecoderLayerState,
    Qwen38TTNNLayerNamespace,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.model import (
    MAX_CONTEXT,
    LayerObserver,
    Qwen38TTNNGreedyStep,
    Qwen38TTNNTextModel,
    Qwen38TTNNTextModelOutput,
    Qwen38TTNNTextModelState,
)

PHYSICAL_IDS = (10, 11, 12, 13)


class _FakeMesh:
    shape = (1, 4)

    def get_num_devices(self) -> int:
        return 4

    def get_device_ids(self) -> list[int]:
        return list(PHYSICAL_IDS)


class _Clock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        self.value += 10
        return self.value


class _FailingClock(_Clock):
    def __init__(self, *, fail_on: int) -> None:
        super().__init__()
        self.calls = 0
        self.fail_on = fail_on

    def __call__(self) -> int:
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("injected clock failure")
        return super().__call__()


class _Synchronizer:
    def __init__(self, *, fail_on: int | None = None) -> None:
        self.calls = 0
        self.fail_on = fail_on

    def __call__(self) -> None:
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("injected synchronize failure")


def _layer_states(position: int, owner: object) -> tuple[Qwen38TTNNDecoderLayerState, ...]:
    del owner
    return tuple(
        Qwen38TTNNDecoderLayerState(
            namespace=Qwen38TTNNLayerNamespace.BACKBONE,
            layer_index=index,
            position=position,
            attention=object(),
            ple=None,
        )
        for index in range(BACKBONE_LAYERS)
    )


_AUX = tuple(Qwen38TTNNDecoderLayerAux(routing=None, selection=None, reused_qsa_selection=False) for _ in range(48))


class _FakeEngine:
    def __init__(self, next_tokens: tuple[int, ...]) -> None:
        self.next_tokens = deque(next_tokens)
        self.owner = object()
        self.inputs: list[tuple[str, int, int]] = []
        self.allocate_calls = 0
        self.reset_calls = 0
        self.release_calls = 0
        self.model_calls = 0
        self.fail_on_model_call: int | None = None
        self.fail_release = False
        self.return_bad_state = False
        self.return_foreign_state = False
        self.layer_observers = []

    def state(self, position: int) -> Qwen38TTNNTextModelState:
        return Qwen38TTNNTextModelState(position, _layer_states(position, self.owner), self.owner)

    def allocate_state(self) -> Qwen38TTNNTextModelState:
        self.allocate_calls += 1
        return self.state(0)

    def reset_state(self, state: Qwen38TTNNTextModelState) -> Qwen38TTNNTextModelState:
        assert state.position >= 0
        self.reset_calls += 1
        return self.state(0)

    def release_state(self, state: Qwen38TTNNTextModelState) -> None:
        assert state.position >= 0
        self.release_calls += 1
        if self.fail_release:
            raise RuntimeError("injected state release failure")

    def _before_call(self) -> None:
        self.model_calls += 1
        if self.model_calls == self.fail_on_model_call:
            raise RuntimeError("injected model failure")

    def forward_decode(
        self,
        token_id: int,
        state: Qwen38TTNNTextModelState,
        **kwargs,
    ) -> Qwen38TTNNTextModelOutput:
        self._before_call()
        self.layer_observers.append(kwargs.pop("layer_observer"))
        assert kwargs == {
            "return_logits": False,
            "resolve_greedy": False,
            "retain_hidden": False,
            "retain_hyper_residual": False,
            "retain_input_state": False,
            "return_routing": False,
        }
        self.inputs.append(("forward", token_id, state.position))
        next_state = self.state(state.position + 1)
        return Qwen38TTNNTextModelOutput(
            input_token_id=token_id,
            position=state.position,
            hyper_residual_sharded=None,
            hidden_sharded=None,
            logits=None,
            greedy_token=None,
            state=next_state,
            layer_aux=_AUX,
        )

    def greedy_step(
        self,
        token_id: int,
        state: Qwen38TTNNTextModelState,
        **kwargs,
    ) -> Qwen38TTNNGreedyStep:
        self._before_call()
        self.layer_observers.append(kwargs.pop("layer_observer"))
        assert kwargs == {"retain_input_state": False}
        self.inputs.append(("greedy", token_id, state.position))
        if not self.next_tokens:
            raise RuntimeError("fake greedy token stream exhausted")
        if self.return_bad_state:
            next_state = object()
        elif self.return_foreign_state:
            next_state = Qwen38TTNNTextModelState(
                state.position + 1,
                _layer_states(state.position + 1, object()),
                object(),
            )
        else:
            next_state = self.state(state.position + 1)
        return Qwen38TTNNGreedyStep(
            input_token_id=token_id,
            next_token_id=self.next_tokens.popleft(),
            state=next_state,  # type: ignore[arg-type]
            layer_aux=_AUX,
        )


def _provenance() -> Qwen38BuildProvenance:
    return Qwen38BuildProvenance(
        checkpoint_revision=PINNED_CHECKPOINT_REVISION,
        checkpoint_index_sha256=INDEX_SHA256,
        checkpoint_config_sha256=CONFIG_SHA256,
        checkpoint_file_manifest_sha256=CHECKPOINT_FILE_MANIFEST_SHA256,
        checkpoint_hash_manifest_sha256=PINNED_TENSOR_MANIFEST_SHA256,
        tt_metal_sha="1" * 40,
        ttnn_runtime_sha256="2" * 64,
    )


def _target(
    next_tokens: tuple[int, ...],
    *,
    allocated_context: int = MAX_CONTEXT,
) -> tuple[Qwen38TTNNBuiltTarget, _FakeEngine, Qwen38BuildProvenance]:
    provenance = _provenance()
    identity = Qwen38LiveBuildIdentity(
        provenance=provenance,
        mesh_shape=(1, 4),
        physical_ids=PHYSICAL_IDS,
        collective_topology="Ring",
        dram_bank_ring_order=(6, 5, 4, 3, 2, 1, 0),
        ring_size=7,
        qsa_cache_capacity=allocated_context,
    )
    mesh = _FakeMesh()
    contract = __import__(
        "models.demos.blackhole.qwen38_flash_next.ttnn.contracts", fromlist=["Qwen38MeshContract"]
    ).Qwen38MeshContract(PHYSICAL_IDS)
    streamer = object()
    layers = tuple(SimpleNamespace(expert_streamer=streamer) for _ in range(BACKBONE_LAYERS))
    model_io = object()
    final_mixer = object()
    model = object.__new__(Qwen38TTNNTextModel)
    model.mesh_device = mesh
    model.mesh_contract = contract
    model.model_io = model_io
    model.layers = layers
    model.final_mixer = final_mixer
    model._poisoned_error = None
    model._active_snapshot = None
    model._runtime_owner = None
    model.allocated_context = allocated_context
    engine = _FakeEngine(next_tokens)
    model._state_owner = engine.owner
    model.allocate_state = engine.allocate_state
    model.reset_state = engine.reset_state
    model.release_state = engine.release_state
    model.forward_decode = engine.forward_decode
    model.greedy_step = engine.greedy_step
    components = Qwen38TTNNTargetComponents(
        identity=identity,
        bf4_cache=object(),
        io_cache=object(),
        expert_streamer=streamer,
        model_io=model_io,
        layers=layers,
        final_mixer=final_mixer,
    )
    return Qwen38TTNNBuiltTarget(model=model, components=components), engine, provenance


def _session(
    next_tokens: tuple[int, ...],
    *,
    stop_on_eos: bool = True,
    synchronizer: _Synchronizer | None = None,
    clock: _Clock | None = None,
    layer_observer: LayerObserver | None = None,
    allocated_context: int = MAX_CONTEXT,
) -> tuple[Qwen38OrdinaryDecodeSession, _FakeEngine, _Synchronizer]:
    target, engine, provenance = _target(next_tokens, allocated_context=allocated_context)
    sync = synchronizer or _Synchronizer()
    session = Qwen38OrdinaryDecodeSession(
        target,
        expected_provenance=provenance,
        expected_physical_ids=PHYSICAL_IDS,
        expected_identity_key=target.components.identity.key,
        eos_token_ids=(2,),
        stop_on_eos=stop_on_eos,
        layer_observer=layer_observer,
        synchronize=sync,
        clock_ns=clock or _Clock(),
    )
    return session, engine, sync


def test_decode_static_contract_and_live_identity_are_exact() -> None:
    validate_decode_static_contract()
    session, engine, _ = _session((2,))
    assert engine.allocate_calls == 1
    assert session.identity.physical_ids == PHYSICAL_IDS
    assert session.identity.mesh_shape == (1, 4)
    assert session.identity.ring_size == 7
    assert sorted(session.identity.dram_bank_ring_order) == list(range(session.identity.ring_size))
    assert session.identity.provenance == _provenance()
    assert session.allocated_context == MAX_CONTEXT == 262_144
    session.close()


def test_serial_prompt_and_decode_track_pending_token_and_timing_domains() -> None:
    session, engine, sync = _session((10, 11, 2))

    first = session.begin(torch.tensor([[5, 6]], dtype=torch.int64), max_new_tokens=4)
    assert (first.token_id, first.token_index, first.cache_position, first.stop_reason) == (10, 0, 2, None)
    assert first.timing.phase is Qwen38OrdinaryTimingPhase.TTFT
    assert first.timing.input_token_count == 2
    assert [item.input_position for item in first.timing.invocations] == [0, 1]
    assert first.timing.model_call_ns >= 0
    assert first.timing.explicit_sync_ns >= 0
    assert first.timing.host_overhead_ns >= 0
    assert session.pending_token_id == 10

    second = session.step()
    third = session.step()
    assert (second.token_id, second.cache_position, second.stop_reason) == (11, 3, None)
    assert second.timing.phase is Qwen38OrdinaryTimingPhase.DECODE
    assert (third.token_id, third.cache_position, third.stop_reason) == (2, 4, Qwen38OrdinaryStopReason.EOS)
    assert session.status is Qwen38OrdinarySessionStatus.FINISHED
    assert session.pending_token_id == 2
    assert engine.inputs == [
        ("forward", 5, 0),
        ("greedy", 6, 1),
        ("greedy", 10, 2),
        ("greedy", 11, 3),
    ]
    assert sync.calls == 9  # allocation drain plus two boundaries per target invocation
    assert session.metrics == (first, second, third)


def test_layer_observer_is_propagated_to_every_model_invocation() -> None:
    observed = []

    def observer(layer_index, residual, state, aux) -> None:
        observed.append((layer_index, residual, state, aux))

    session, engine, _ = _session((10, 2), layer_observer=observer)
    session.begin([4], max_new_tokens=2)
    session.step()

    assert engine.layer_observers == [observer, observer]
    assert observed == []  # The fake engine records propagation; real layers own callback invocation.


def test_max_new_tokens_and_context_cache_continuation_are_explicit(expect_error) -> None:
    session, engine, _ = _session((10, 11))
    first = session.begin([7], max_new_tokens=1)
    assert first.stop_reason is Qwen38OrdinaryStopReason.MAX_NEW_TOKENS
    assert session.status is Qwen38OrdinarySessionStatus.FINISHED

    with expect_error(ValueError, "continuation must first consume pending token 10"):
        session.begin([9], max_new_tokens=1)
    assert session.pending_token_id == 10
    assert session.state_position == 1

    second = session.begin([10, 20], max_new_tokens=1)
    assert second.token_id == 11
    assert second.cache_position == 3
    assert engine.inputs[-2:] == [("forward", 10, 1), ("greedy", 20, 2)]


def test_allocated_context_stops_decode_and_rejects_prompt_before_model_mutation(expect_error) -> None:
    capacity = RESIDENT_MAX_QSA_CACHE_CAPACITY
    session, engine, _ = _session((10,), allocated_context=capacity)
    session._state = engine.state(capacity - 1)

    event = session.begin([7], max_new_tokens=4)
    assert event.cache_position == capacity
    assert event.stop_reason is Qwen38OrdinaryStopReason.CONTEXT_LENGTH
    assert session.status is Qwen38OrdinarySessionStatus.FINISHED
    assert engine.model_calls == 1
    with expect_error(Qwen38OrdinaryDecodeError, "status is finished"):
        session.step()
    assert engine.model_calls == 1

    other, other_engine, _ = _session((11,), allocated_context=capacity)
    other._state = other_engine.state(capacity - 1)
    with expect_error(ValueError, "allocated context 32768"):
        other.begin([7, 8], max_new_tokens=1)
    assert other_engine.model_calls == 0
    assert other.status is Qwen38OrdinarySessionStatus.IDLE


def test_allocated_context_must_match_authenticated_build_identity_before_state_allocation(expect_error) -> None:
    target, engine, provenance = _target((2,), allocated_context=RESIDENT_MAX_QSA_CACHE_CAPACITY)
    target.model.allocated_context = MAX_CONTEXT
    with expect_error(ValueError, "differs from its authenticated live build identity"):
        Qwen38OrdinaryDecodeSession(
            target,
            expected_provenance=provenance,
            expected_physical_ids=PHYSICAL_IDS,
            expected_identity_key=target.components.identity.key,
            eos_token_ids=(2,),
            synchronize=_Synchronizer(),
        )
    assert engine.allocate_calls == 0


def test_generate_is_greedy_serial_and_stops_at_eos() -> None:
    session, _, _ = _session((8, 9, 2, 99))
    events = list(session.generate([4], max_new_tokens=8))
    assert [event.token_id for event in events] == [8, 9, 2]
    assert events[-1].stop_reason is Qwen38OrdinaryStopReason.EOS


def test_benchmark_eos_bypass_preserves_max_token_and_context_stops(expect_error) -> None:
    session, _, _ = _session((2, 8, 9), stop_on_eos=False)
    first = session.begin([4], max_new_tokens=3)
    assert first.token_id == 2
    assert first.stop_reason is None
    assert session.stop_on_eos is False
    second = session.step()
    third = session.step()
    assert second.stop_reason is None
    assert third.stop_reason is Qwen38OrdinaryStopReason.MAX_NEW_TOKENS
    assert session.status is Qwen38OrdinarySessionStatus.FINISHED

    capacity = RESIDENT_MAX_QSA_CACHE_CAPACITY
    bounded, engine, _ = _session((2,), stop_on_eos=False, allocated_context=capacity)
    bounded._state = engine.state(capacity - 1)
    event = bounded.begin([7], max_new_tokens=4)
    assert event.token_id == 2
    assert event.stop_reason is Qwen38OrdinaryStopReason.CONTEXT_LENGTH
    with expect_error(Qwen38OrdinaryDecodeError, "status is finished"):
        bounded.step()


@pytest.mark.parametrize("invalid", (None, 0, 1, "false", object()))
def test_stop_on_eos_requires_an_exact_bool_before_state_allocation(invalid, expect_error) -> None:
    target, engine, provenance = _target((2,))
    with expect_error(TypeError, "stop_on_eos must be an explicit bool"):
        Qwen38OrdinaryDecodeSession(
            target,
            expected_provenance=provenance,
            expected_physical_ids=PHYSICAL_IDS,
            expected_identity_key=target.components.identity.key,
            eos_token_ids=(2,),
            stop_on_eos=invalid,
            synchronize=_Synchronizer(),
        )
    assert engine.allocate_calls == 0


def test_invalid_global_batch_and_flatness_fail_before_state_mutation(expect_error) -> None:
    session, engine, _ = _session((2,))
    with expect_error(ValueError, "true-global-B1"):
        session.begin(torch.tensor([[1], [2]], dtype=torch.int64), max_new_tokens=1)
    with expect_error(TypeError, "flat sequence"):
        session.begin([[1]], max_new_tokens=1)  # type: ignore[list-item]
    assert engine.model_calls == 0
    assert session.status is Qwen38OrdinarySessionStatus.IDLE


def test_model_failure_poison_is_terminal_and_skips_release(expect_error) -> None:
    session, engine, _ = _session((2,))
    engine.fail_on_model_call = 1
    with expect_error(Qwen38OrdinaryDecodePoisonedError, "terminate the task-owned process"):
        session.begin([4], max_new_tokens=1)
    raised = session.poison_error
    assert raised is not None
    assert raised.operation == "greedy_step"
    assert session.status is Qwen38OrdinarySessionStatus.POISONED
    with expect_error(Qwen38OrdinaryDecodePoisonedError):
        session.close()
    assert session.poison_error is raised
    assert engine.release_calls == 0


def test_post_call_synchronize_failure_poison_is_terminal(expect_error) -> None:
    sync = _Synchronizer(fail_on=3)
    session, engine, _ = _session((2,), synchronizer=sync)
    with expect_error(Qwen38OrdinaryDecodePoisonedError, "synchronize failure"):
        session.begin([4], max_new_tokens=1)
    assert engine.model_calls == 1
    assert session.status is Qwen38OrdinarySessionStatus.POISONED
    assert engine.release_calls == 0


def test_wrong_state_owner_type_poison_is_terminal(expect_error) -> None:
    session, engine, _ = _session((2,))
    engine.return_bad_state = True
    with expect_error(Qwen38OrdinaryDecodePoisonedError, "non-Qwen38 TTNN model state"):
        session.begin([4], max_new_tokens=1)
    assert session.status is Qwen38OrdinarySessionStatus.POISONED


def test_foreign_exact_state_owner_poison_is_terminal(expect_error) -> None:
    session, engine, _ = _session((2,))
    engine.return_foreign_state = True
    with expect_error(Qwen38OrdinaryDecodePoisonedError, "different text-model owner"):
        session.begin([4], max_new_tokens=1)
    assert session.status is Qwen38OrdinarySessionStatus.POISONED


@pytest.mark.parametrize(("fail_on", "operation"), ((3, "begin"), (4, "greedy_step")))
def test_clock_failure_after_admission_poison_is_terminal(fail_on: int, operation: str, expect_error) -> None:
    session, engine, _ = _session((2,), clock=_FailingClock(fail_on=fail_on))
    with expect_error(Qwen38OrdinaryDecodePoisonedError, "clock failure"):
        session.begin([4], max_new_tokens=1)
    assert session.poison_error is not None
    assert session.poison_error.operation == operation
    assert session.status is Qwen38OrdinarySessionStatus.POISONED
    assert engine.release_calls == 0


def test_wrong_live_identity_fails_before_state_allocation(expect_error) -> None:
    target, engine, provenance = _target((2,))
    with expect_error(ValueError, "differs from expected"):
        Qwen38OrdinaryDecodeSession(
            target,
            expected_provenance=provenance,
            expected_physical_ids=PHYSICAL_IDS,
            expected_identity_key="0" * 64,
            eos_token_ids=(2,),
            synchronize=_Synchronizer(),
        )
    assert engine.allocate_calls == 0


def test_wrong_independently_supplied_provenance_fails_before_state_allocation(expect_error) -> None:
    target, engine, provenance = _target((2,))
    wrong_provenance = replace(provenance, ttnn_runtime_sha256="3" * 64)
    with expect_error(ValueError, "provenance differs"):
        Qwen38OrdinaryDecodeSession(
            target,
            expected_provenance=wrong_provenance,
            expected_physical_ids=PHYSICAL_IDS,
            expected_identity_key=target.components.identity.key,
            eos_token_ids=(2,),
            synchronize=_Synchronizer(),
        )
    assert engine.allocate_calls == 0


def test_reset_and_close_own_state_once(expect_error) -> None:
    session, engine, sync = _session((7, 2))
    session.begin([4], max_new_tokens=1)
    session.reset()
    assert session.status is Qwen38OrdinarySessionStatus.IDLE
    assert session.state_position == 0
    assert session.pending_token_id is None
    assert engine.reset_calls == 1
    session.close()
    session.close()
    assert engine.release_calls == 1
    assert session.status is Qwen38OrdinarySessionStatus.CLOSED
    assert sync.calls >= 1
    with expect_error(Qwen38OrdinaryDecodeError, "closed"):
        session.begin([4], max_new_tokens=1)


def test_close_failure_poison_is_not_retried(expect_error) -> None:
    session, engine, _ = _session((2,))
    engine.fail_release = True
    with expect_error(Qwen38OrdinaryDecodePoisonedError, "state release failure"):
        session.close()
    raised = session.poison_error
    assert raised is not None
    assert raised.operation == "release_state"
    assert session.status is Qwen38OrdinarySessionStatus.POISONED
    with expect_error(Qwen38OrdinaryDecodePoisonedError):
        session.close()
    assert session.poison_error is raised
    assert engine.release_calls == 1


def test_from_builder_validates_before_building_and_builds_once() -> None:
    target, engine, provenance = _target((2,))
    builder = object.__new__(Qwen38TTNNBuilder)
    builder.provenance = provenance
    builder.identity = target.components.identity
    build_calls: list[bool] = []

    def build_target() -> Qwen38TTNNBuiltTarget:
        build_calls.append(True)
        return target

    builder.build_target = build_target
    session = Qwen38OrdinaryDecodeSession.from_builder(
        builder,
        expected_provenance=provenance,
        expected_physical_ids=PHYSICAL_IDS,
        expected_identity_key=target.components.identity.key,
        eos_token_ids=(2,),
        synchronize=_Synchronizer(),
        clock_ns=_Clock(),
    )
    assert build_calls == [True]
    assert engine.allocate_calls == 1
    session.close()


def test_from_builder_rejects_invalid_session_arguments_before_building(expect_error) -> None:
    target, _, provenance = _target((2,))
    builder = object.__new__(Qwen38TTNNBuilder)
    builder.provenance = provenance
    builder.identity = target.components.identity
    build_calls: list[bool] = []

    def build_target() -> Qwen38TTNNBuiltTarget:
        build_calls.append(True)
        return target

    builder.build_target = build_target
    common = {
        "expected_provenance": provenance,
        "expected_physical_ids": PHYSICAL_IDS,
        "expected_identity_key": target.components.identity.key,
    }

    with expect_error(ValueError, "at least one pinned EOS token ID"):
        Qwen38OrdinaryDecodeSession.from_builder(builder, eos_token_ids=(), **common)
    with expect_error(TypeError, "synchronize must be callable"):
        Qwen38OrdinaryDecodeSession.from_builder(builder, eos_token_ids=(2,), synchronize=object(), **common)
    with expect_error(TypeError, "clock_ns must be callable"):
        Qwen38OrdinaryDecodeSession.from_builder(builder, eos_token_ids=(2,), clock_ns=object(), **common)
    with expect_error(TypeError, "layer_observer must be callable"):
        Qwen38OrdinaryDecodeSession.from_builder(builder, eos_token_ids=(2,), layer_observer=object(), **common)
    with expect_error(TypeError, "stop_on_eos must be an explicit bool"):
        Qwen38OrdinaryDecodeSession.from_builder(builder, eos_token_ids=(2,), stop_on_eos=0, **common)
    with expect_error(ValueError, "provenance differs"):
        Qwen38OrdinaryDecodeSession.from_builder(
            builder,
            eos_token_ids=(2,),
            **{**common, "expected_provenance": replace(provenance, ttnn_runtime_sha256="3" * 64)},
        )
    with expect_error(ValueError, "live identity differs"):
        Qwen38OrdinaryDecodeSession.from_builder(
            builder,
            eos_token_ids=(2,),
            **{**common, "expected_physical_ids": (10, 11, 12, 14)},
        )
    with expect_error(ValueError, "live identity differs"):
        Qwen38OrdinaryDecodeSession.from_builder(
            builder,
            eos_token_ids=(2,),
            **{**common, "expected_identity_key": "0" * 64},
        )

    assert build_calls == []


def test_from_builder_freezes_mutable_eos_before_target_build() -> None:
    target, _, provenance = _target((2,))
    builder = object.__new__(Qwen38TTNNBuilder)
    builder.provenance = provenance
    builder.identity = target.components.identity
    eos_ids = [2]

    def build_target() -> Qwen38TTNNBuiltTarget:
        eos_ids.clear()
        return target

    builder.build_target = build_target
    session = Qwen38OrdinaryDecodeSession.from_builder(
        builder,
        expected_provenance=provenance,
        expected_physical_ids=PHYSICAL_IDS,
        expected_identity_key=target.components.identity.key,
        eos_token_ids=eos_ids,
        synchronize=_Synchronizer(),
        clock_ns=_Clock(),
    )
    assert session.eos_token_ids == (2,)
    session.close()


def test_one_target_has_one_live_session_owner_and_clean_close_releases_claim(expect_error) -> None:
    target, engine, provenance = _target((2, 2))
    common = {
        "expected_provenance": provenance,
        "expected_physical_ids": PHYSICAL_IDS,
        "expected_identity_key": target.components.identity.key,
        "eos_token_ids": (2,),
        "synchronize": _Synchronizer(),
        "clock_ns": _Clock(),
    }
    first = Qwen38OrdinaryDecodeSession(target, **common)
    with expect_error(RuntimeError, "already has a live runtime owner"):
        Qwen38OrdinaryDecodeSession(target, **common)
    assert engine.allocate_calls == 1

    first.close()
    second = Qwen38OrdinaryDecodeSession(target, **common)
    assert engine.allocate_calls == 2
    second.close()


def test_poisoned_session_retains_exclusive_target_claim(expect_error) -> None:
    target, engine, provenance = _target((2,))
    common = {
        "expected_provenance": provenance,
        "expected_physical_ids": PHYSICAL_IDS,
        "expected_identity_key": target.components.identity.key,
        "eos_token_ids": (2,),
        "synchronize": _Synchronizer(),
        "clock_ns": _Clock(),
    }
    session = Qwen38OrdinaryDecodeSession(target, **common)
    engine.fail_on_model_call = 1
    with expect_error(Qwen38OrdinaryDecodePoisonedError):
        session.begin([4], max_new_tokens=1)
    with expect_error(RuntimeError, "already has a live runtime owner"):
        Qwen38OrdinaryDecodeSession(target, **common)
    assert engine.allocate_calls == 1


def test_timing_excludes_session_validation_and_output_release(monkeypatch) -> None:
    clock = _Clock()
    original_release = Qwen38TTNNTextModelOutput.release_tensors

    def delayed_release(output) -> None:
        clock.value += 1_000
        original_release(output)

    monkeypatch.setattr(Qwen38TTNNTextModelOutput, "release_tensors", delayed_release)
    session, _, _ = _session((2,), clock=clock)
    event = session.begin([4, 5], max_new_tokens=1)
    first = event.timing.invocations[0]
    assert first.model_call_ns == 10
    assert first.host_overhead_ns >= 1_000
    session.close()


def test_constructor_reset_and_close_sync_failures_are_terminal(expect_error) -> None:
    target, engine, provenance = _target((2,))
    common = {
        "expected_provenance": provenance,
        "expected_physical_ids": PHYSICAL_IDS,
        "expected_identity_key": target.components.identity.key,
        "eos_token_ids": (2,),
        "clock_ns": _Clock(),
    }
    with expect_error(Qwen38OrdinaryDecodePoisonedError, "allocate_state"):
        Qwen38OrdinaryDecodeSession(target, synchronize=_Synchronizer(fail_on=1), **common)
    assert engine.allocate_calls == 1 and engine.release_calls == 0

    target, engine, provenance = _target((2,))
    common["expected_provenance"] = provenance
    common["expected_identity_key"] = target.components.identity.key
    reset_session = Qwen38OrdinaryDecodeSession(target, synchronize=_Synchronizer(fail_on=2), **common)
    with expect_error(Qwen38OrdinaryDecodePoisonedError, "reset_state"):
        reset_session.reset()
    assert engine.reset_calls == 1 and engine.release_calls == 0

    target, engine, provenance = _target((2,))
    common["expected_provenance"] = provenance
    common["expected_identity_key"] = target.components.identity.key
    close_session = Qwen38OrdinaryDecodeSession(target, synchronize=_Synchronizer(fail_on=2), **common)
    with expect_error(Qwen38OrdinaryDecodePoisonedError, "release_state"):
        close_session.close()
    with expect_error(Qwen38OrdinaryDecodePoisonedError):
        close_session.close()
    assert engine.release_calls == 1

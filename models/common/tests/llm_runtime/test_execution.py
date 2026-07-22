# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import inspect
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import models.common.llm_runtime.execution as execution_module
import ttnn
from models.common.llm_runtime.decode import DecodeRuntime
from models.common.llm_runtime.decode import InvocationResult as DecodeInvocationResult
from models.common.llm_runtime.execution import EagerExecutor, TracedExecutor
from models.common.llm_runtime.prefill import InvocationResult as PrefillInvocationResult
from models.common.llm_runtime.prefill import PrefillRuntime
from models.common.llm_runtime.program_compiler import ProgramCompiler
from models.common.llm_runtime.trace_compiler import TraceCompiler


@dataclass(frozen=True)
class _Signature:
    operation: str
    variant: int

    @property
    def key_material(self):
        return (("operation", self.operation), ("variant", self.variant))


def _runtime(runtime_type, **methods):
    runtime = object.__new__(runtime_type)
    for name, method in methods.items():
        setattr(runtime, name, method)
    return runtime


def _compiler(monkeypatch):
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: None)
    return ProgramCompiler("mesh", lambda: object())


def _trace_compiler(program_compiler, *, mode="all"):
    trace_config = SimpleNamespace(
        mode=mode,
        prefill_enabled=mode == "all",
        decode_enabled=mode in ("decode_only", "all"),
    )
    return TraceCompiler(program_compiler, "mesh", trace_config)


def _prepared_prefill(*, trace_eligible=True, signatures=None, name="regular"):
    if signatures is None:
        signatures = (_Signature("prefill", 1),)
    return SimpleNamespace(
        name=name,
        program_signatures=signatures,
        trace_eligible=trace_eligible,
    )


def _prepared_decode(*, variant=1):
    return SimpleNamespace(
        variant=variant,
        device_feedback=True,
        reset_batch=False,
        page_table_changed=False,
        sampling_params=None,
    )


def test_execution_strategies_use_exact_identity_composition_without_type_frameworks(monkeypatch):
    prefill = _runtime(PrefillRuntime)
    decode = _runtime(DecodeRuntime)
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(prefill=prefill, decode=decode, program_compiler=program_compiler)
    trace_compiler = _trace_compiler(program_compiler)
    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler)

    assert eager.prefill is prefill
    assert eager.decode is decode
    assert eager.program_compiler is program_compiler
    assert traced.eager is eager
    assert traced.prefill is prefill
    assert traced.decode is decode
    assert traced.program_compiler is program_compiler
    assert traced.trace_compiler is trace_compiler
    assert EagerExecutor not in TracedExecutor.__mro__
    assert EagerExecutor.__bases__ == (object,)
    assert TracedExecutor.__bases__ == (object,)

    source = inspect.getsource(execution_module)
    assert "Protocol" not in source
    assert "ABC" not in source
    assert "LightweightModule" not in source
    assert not hasattr(execution_module, "EagerExecutorConfig")
    assert not hasattr(execution_module, "TracedExecutorConfig")
    assert not hasattr(EagerExecutor, "cleanup")
    assert not hasattr(TracedExecutor, "cleanup")


def test_traced_constructor_rejects_a_different_program_compiler(monkeypatch, expect_error):
    eager = EagerExecutor(
        prefill=_runtime(PrefillRuntime),
        decode=_runtime(DecodeRuntime),
        program_compiler=_compiler(monkeypatch),
    )
    unrelated_trace_compiler = _trace_compiler(_compiler(monkeypatch))

    with expect_error(ValueError, "compose eager.program_compiler"):
        TracedExecutor(eager=eager, trace_compiler=unrelated_trace_compiler)


def test_eager_prefill_prepares_once_and_compiles_all_signatures_from_same_object(monkeypatch):
    prepared = _prepared_prefill(
        signatures=(_Signature("prefill", 1), _Signature("prefill", 2)),
    )
    prepared_seen = []
    prepare_calls = []
    prefill = _runtime(
        PrefillRuntime,
        prepare=lambda **kwargs: prepare_calls.append(kwargs) or (prepared,),
        invoke=lambda request: prepared_seen.append(request) or PrefillInvocationResult(torch.zeros(1), ()),
    )
    eager = EagerExecutor(prefill=prefill, decode=_runtime(DecodeRuntime), program_compiler=_compiler(monkeypatch))

    eager.compile_prefill(tokens=torch.zeros(1, 1))

    assert len(prepare_calls) == 1
    assert prepared_seen == [prepared, prepared]
    assert len(eager.program_compiler.programs) == 2


def test_traced_prefill_compile_registers_capture_from_the_same_prepared_object(monkeypatch):
    prepared = _prepared_prefill()
    identity_events = []
    operation_plan = SimpleNamespace(
        signature=_Signature("prefill-trace", 1),
        prepare_inputs=lambda: (),
        capture=lambda persistent: torch.zeros(1),
        refresh_fields=("tokens",),
    )
    prefill = _runtime(
        PrefillRuntime,
        prepare=lambda **kwargs: identity_events.append(("prepare", prepared)) or (prepared,),
        invoke=lambda request: identity_events.append(("invoke", request))
        or PrefillInvocationResult(torch.zeros(1), ()),
        capture_plan=lambda request: identity_events.append(("capture_plan", request)) or operation_plan,
    )
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(prefill=prefill, decode=_runtime(DecodeRuntime), program_compiler=program_compiler)
    trace_compiler = _trace_compiler(program_compiler)
    registered = []
    trace_compiler.register_capture_plan = registered.append
    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler)

    traced.compile_prefill(tokens=torch.zeros(1, 1))

    assert [event[0] for event in identity_events] == ["prepare", "invoke", "capture_plan"]
    assert all(event[1] is prepared for event in identity_events)
    assert len(registered) == 1
    assert registered[0].program_key in program_compiler.programs


def test_traced_prefill_recompile_reuses_existing_trace_association(monkeypatch):
    prepared = _prepared_prefill()
    prefill = _runtime(
        PrefillRuntime,
        invoke=lambda request: PrefillInvocationResult(torch.zeros(1), ()),
        capture_plan=lambda request: (_ for _ in ()).throw(AssertionError("capture plan rebuilt")),
    )
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(prefill=prefill, decode=_runtime(DecodeRuntime), program_compiler=program_compiler)
    trace_compiler = _trace_compiler(program_compiler)
    trace_compiler.trace_key_for_program = lambda program_key: "existing-trace"
    trace_compiler.register_capture_plan = lambda plan: (_ for _ in ()).throw(AssertionError("plan registered"))
    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler)

    traced.compile_prefill_prepared(prepared, enable_trace=True)


def test_traced_decode_recompile_reuses_existing_trace_association(monkeypatch):
    prepared = _prepared_decode()
    decode = _runtime(
        DecodeRuntime,
        program_signature=lambda request: _Signature("decode", 1),
        invoke=lambda request, **kwargs: DecodeInvocationResult(torch.zeros(1), (), False),
        capture_plan=lambda request: (_ for _ in ()).throw(AssertionError("capture plan rebuilt")),
    )
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(prefill=_runtime(PrefillRuntime), decode=decode, program_compiler=program_compiler)
    trace_compiler = _trace_compiler(program_compiler)
    trace_compiler.trace_key_for_program = lambda program_key: "existing-trace"
    trace_compiler.register_capture_plan = lambda plan: (_ for _ in ()).throw(AssertionError("plan registered"))
    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler)

    traced.compile_decode_prepared(prepared, enable_trace=True)


@pytest.mark.parametrize("name", ["cached-prefill-one-chunk", "multi-chunk-prefill"])
def test_trace_ineligible_prefill_falls_back_to_eager_exactly_once(monkeypatch, name):
    prepared = _prepared_prefill(trace_eligible=False, name=name)
    calls = []
    prefill = _runtime(
        PrefillRuntime,
        prepare=lambda **kwargs: calls.append(("prepare", prepared)) or (prepared,),
        invoke=lambda request: calls.append(("invoke", request)) or PrefillInvocationResult(name, ()),
        assemble=lambda results, **kwargs: calls.append(("assemble", results[0][0])) or results[0][1].value,
    )
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(prefill=prefill, decode=_runtime(DecodeRuntime), program_compiler=program_compiler)
    trace_compiler = _trace_compiler(program_compiler)
    trace_compiler.replay = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("trace replayed"))
    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler)

    result = traced.prefill_forward(tokens=torch.zeros(1, 1))

    assert result == name
    assert calls == [("prepare", prepared), ("invoke", prepared), ("assemble", prepared)]


def test_decode_only_and_explicit_eager_prefill_delegate_without_replay(monkeypatch, expect_error):
    prepared = _prepared_prefill(trace_eligible=True)
    invocations = []
    prefill = _runtime(
        PrefillRuntime,
        prepare=lambda **kwargs: (prepared,),
        invoke=lambda request: invocations.append(request) or PrefillInvocationResult("eager", ()),
        assemble=lambda results, **kwargs: results[0][1].value,
    )
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(prefill=prefill, decode=_runtime(DecodeRuntime), program_compiler=program_compiler)
    trace_compiler = _trace_compiler(program_compiler, mode="decode_only")
    trace_compiler.replay = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("trace replayed"))
    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler)

    assert traced.prefill_forward(tokens=torch.zeros(1, 1)) == "eager"
    with expect_error(ValueError, "disagrees with static prefill trace policy"):
        traced.prefill_forward(tokens=torch.zeros(1, 1), enable_trace=True)

    trace_compiler.trace_config.prefill_enabled = True
    assert traced.prefill_forward(tokens=torch.zeros(1, 1), enable_trace=False) == "eager"
    assert invocations == [prepared, prepared]


def test_prefill_replay_refresh_and_finish_use_the_same_prepared_object(monkeypatch):
    prepared = _prepared_prefill(trace_eligible=True)
    persistent = object()
    hidden = object()
    identity_events = []
    prefill = _runtime(
        PrefillRuntime,
        prepare=lambda **kwargs: identity_events.append(("prepare", prepared)) or (prepared,),
        refresh_trace=lambda request, values: identity_events.append(("refresh", request, values)),
        finish_trace=lambda request, value, values: identity_events.append(("finish", request, value, values))
        or "traced",
        assemble=lambda results, **kwargs: results[0][1],
    )
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(prefill=prefill, decode=_runtime(DecodeRuntime), program_compiler=program_compiler)
    trace_compiler = _trace_compiler(program_compiler)
    artifact = SimpleNamespace(persistent_inputs=SimpleNamespace(values=persistent))
    record = SimpleNamespace(artifact=artifact)
    trace_compiler.replay = lambda program_key, refresh, **kwargs: refresh(artifact, object()) or hidden
    trace_compiler.trace_key_for_program = lambda program_key: "trace-key"
    trace_compiler.get = lambda trace_key: record
    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler)

    result = traced.prefill_forward(tokens=torch.zeros(1, 1))

    assert result == "traced"
    assert [event[0] for event in identity_events] == ["prepare", "refresh", "finish"]
    assert all(event[1] is prepared for event in identity_events)
    assert identity_events[1][2] is persistent
    assert identity_events[2][2:] == (hidden, persistent)


def test_prefill_missing_trace_artifact_is_an_error_without_eager_reinvocation(monkeypatch, expect_error):
    prepared = _prepared_prefill(trace_eligible=True)
    eager_invocations = []
    prefill = _runtime(
        PrefillRuntime,
        invoke=lambda request: eager_invocations.append(request) or PrefillInvocationResult("eager", ()),
        refresh_trace=lambda request, values: None,
    )
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(prefill=prefill, decode=_runtime(DecodeRuntime), program_compiler=program_compiler)
    trace_compiler = _trace_compiler(program_compiler)
    artifact = SimpleNamespace(persistent_inputs=SimpleNamespace(values=()))
    trace_compiler.replay = lambda program_key, refresh, **kwargs: refresh(artifact, object()) or "hidden"
    trace_compiler.trace_key_for_program = lambda program_key: "missing"
    trace_compiler.get = lambda trace_key: None
    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler)

    with expect_error(RuntimeError, "Required prefill trace"):
        traced.execute_prefill_prepared(prepared, enable_trace=True)

    assert eager_invocations == []


def test_decode_replay_prepares_once_and_uses_same_object_for_refresh_submission_and_consume(monkeypatch):
    prepared = _prepared_decode()
    events = []
    decode = _runtime(
        DecodeRuntime,
        device_feedback_enabled=True,
        prepare=lambda **kwargs: events.append(("prepare", prepared)) or prepared,
        program_signature=lambda request: events.append(("signature", request)) or _Signature("decode", 1),
        refresh_trace=lambda artifact, request, decision: events.append(("refresh", request)),
        note_submitted=lambda request: events.append(("submitted", request)),
        consume=lambda result, **kwargs: events.append(("consume", result)) or result.value,
    )
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(prefill=_runtime(PrefillRuntime), decode=decode, program_compiler=program_compiler)
    trace_compiler = _trace_compiler(program_compiler)
    trace_compiler.replay = lambda program_key, refresh, **kwargs: refresh(object(), object()) or "token"
    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler)

    result = traced.decode_forward(tokens=torch.zeros(1, 1), start_pos=torch.zeros(1), page_table=torch.zeros(1, 1))

    assert result == "token"
    assert [event[0] for event in events] == ["prepare", "signature", "refresh", "submitted", "consume"]
    assert all(event[1] is prepared for event in events[:-1])
    assert isinstance(events[-1][1], DecodeInvocationResult)
    assert events[-1][1].owned is None


def test_explicit_eager_decode_delegates_once_and_execution_objects_do_not_cleanup(monkeypatch):
    prepared = _prepared_decode()
    calls = []
    decode = _runtime(
        DecodeRuntime,
        prepare=lambda **kwargs: calls.append(("prepare", prepared)) or prepared,
        invoke=lambda request, **kwargs: calls.append(("invoke", request))
        or DecodeInvocationResult("eager", (), False),
        consume=lambda result, **kwargs: calls.append(("consume", result.value)) or result.value,
    )
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(prefill=_runtime(PrefillRuntime), decode=decode, program_compiler=program_compiler)
    trace_compiler = _trace_compiler(program_compiler)
    trace_compiler.replay = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("trace replayed"))
    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler)

    assert (
        traced.decode_forward(
            tokens=torch.zeros(1, 1),
            start_pos=torch.zeros(1),
            page_table=torch.zeros(1, 1),
            enable_trace=False,
        )
        == "eager"
    )
    assert calls == [("prepare", prepared), ("invoke", prepared), ("consume", "eager")]
    assert program_compiler.programs == {}
    assert trace_compiler.traces == {}

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import inspect
from dataclasses import dataclass
from types import SimpleNamespace

import torch

import models.common.llm_runtime.execution as execution_module
import ttnn
from models.common.llm_runtime.decode import DecodeRuntime
from models.common.llm_runtime.decode import InvocationResult as DecodeInvocationResult
from models.common.llm_runtime.execution import EagerExecutor, TracedExecutor
from models.common.llm_runtime.prefill.result_collector import InvocationResult as PrefillInvocationResult
from models.common.llm_runtime.prefill.runtime import PrefillRuntime
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
    return TraceCompiler(program_compiler)


def _prepared_prefill(*, trace_eligible=True, signatures=None, name="regular"):
    if signatures is None:
        signatures = (_Signature("prefill", 1),)
    return SimpleNamespace(
        name=name,
        program_signatures=signatures,
        trace_eligible=trace_eligible,
        trace_signature=_Signature("prefill-trace", 1) if trace_eligible else None,
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
    assert traced.eager_executor is eager
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


def test_execution_request_signatures_are_exact_and_aligned():
    required = inspect.Parameter.empty
    positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
    keyword_only = inspect.Parameter.KEYWORD_ONLY
    prefill_contract = [
        ("self", positional, required),
        ("tokens", keyword_only, required),
        ("page_table", keyword_only, required),
        ("prompt_lens", keyword_only, None),
        ("start_pos", keyword_only, None),
        ("empty_slots", keyword_only, None),
        ("sampling_params", keyword_only, None),
    ]
    decode_contract = [
        ("self", positional, required),
        ("tokens", keyword_only, required),
        ("start_pos", keyword_only, required),
        ("page_table", keyword_only, required),
        ("sampling_params", keyword_only, None),
        ("reset_batch", keyword_only, False),
    ]
    decode_forward_contract = [
        *decode_contract,
        ("read_from_device", keyword_only, True),
    ]

    def parameter_contract(method):
        return [
            (parameter.name, parameter.kind, parameter.default)
            for parameter in inspect.signature(method).parameters.values()
        ]

    for executor_type in (EagerExecutor, TracedExecutor):
        assert parameter_contract(executor_type.compile_prefill) == prefill_contract
        assert parameter_contract(executor_type.prefill_forward) == prefill_contract
        assert parameter_contract(executor_type.compile_decode) == decode_contract
        assert parameter_contract(executor_type.decode_forward) == decode_forward_contract

    assert parameter_contract(EagerExecutor._prepare_prefill) == prefill_contract
    assert parameter_contract(EagerExecutor._prepare_decode) == decode_contract
    for method_name in ("compile_prefill", "prefill_forward", "compile_decode", "decode_forward"):
        assert inspect.signature(getattr(EagerExecutor, method_name)) == inspect.signature(
            getattr(TracedExecutor, method_name)
        )


def test_execution_request_methods_reject_kv_cache(expect_error):
    prefill_fields = {
        "tokens": torch.zeros(1, 1),
        "page_table": torch.zeros(1, 1),
    }
    decode_fields = {
        "tokens": torch.zeros(1, 1),
        "start_pos": torch.zeros(1),
        "page_table": torch.zeros(1, 1),
    }

    for executor_type in (EagerExecutor, TracedExecutor):
        executor = object.__new__(executor_type)
        for method_name, fields in (
            ("compile_prefill", prefill_fields),
            ("prefill_forward", prefill_fields),
            ("compile_decode", decode_fields),
            ("decode_forward", decode_fields),
        ):
            with expect_error(TypeError, "kv_cache"):
                getattr(executor, method_name)(**fields, kv_cache=object())


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

    def prepare(
        *,
        tokens,
        page_table,
        prompt_lens=None,
        start_pos=None,
        empty_slots=None,
        sampling_params=None,
    ):
        prepare_calls.append(
            {
                "tokens": tokens,
                "page_table": page_table,
                "prompt_lens": prompt_lens,
                "start_pos": start_pos,
                "empty_slots": empty_slots,
                "sampling_params": sampling_params,
            }
        )
        return (prepared,)

    prefill = _runtime(
        PrefillRuntime,
        prepare=prepare,
        invoke=lambda prepared: prepared_seen.append(prepared) or PrefillInvocationResult(torch.zeros(1), ()),
    )
    eager = EagerExecutor(prefill=prefill, decode=_runtime(DecodeRuntime), program_compiler=_compiler(monkeypatch))
    tokens = torch.zeros(1, 1)
    page_table = torch.zeros(1, 1)
    prompt_lens = torch.tensor([1])
    start_pos = torch.tensor([0])
    empty_slots = [0]
    sampling_params = object()

    eager.compile_prefill(
        tokens=tokens,
        page_table=page_table,
        prompt_lens=prompt_lens,
        start_pos=start_pos,
        empty_slots=empty_slots,
        sampling_params=sampling_params,
    )

    assert prepare_calls == [
        {
            "tokens": tokens,
            "page_table": page_table,
            "prompt_lens": prompt_lens,
            "start_pos": start_pos,
            "empty_slots": empty_slots,
            "sampling_params": sampling_params,
        }
    ]
    assert prepared_seen == [prepared, prepared]


def test_traced_prefill_compile_does_not_interpret_request_eligibility(monkeypatch):
    prepared = _prepared_prefill(trace_eligible=False)
    identity_events = []
    operation_plan = SimpleNamespace(
        signature=_Signature("prefill-trace", 1),
        prepare_inputs=lambda: (),
        capture=lambda persistent: torch.zeros(1),
        refresh_fields=("tokens",),
    )

    def prepare(
        *,
        tokens,
        page_table,
        prompt_lens=None,
        start_pos=None,
        empty_slots=None,
        sampling_params=None,
    ):
        identity_events.append(("prepare", prepared))
        return (prepared,)

    prefill = _runtime(
        PrefillRuntime,
        prepare=prepare,
        invoke=lambda prepared: identity_events.append(("invoke", prepared))
        or PrefillInvocationResult(torch.zeros(1), ()),
        capture_plan=lambda prepared: identity_events.append(("capture_plan", prepared)) or operation_plan,
    )
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(prefill=prefill, decode=_runtime(DecodeRuntime), program_compiler=program_compiler)
    trace_compiler = _trace_compiler(program_compiler)
    registered = []
    trace_compiler.register_capture_plan = registered.append
    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler)

    traced.compile_prefill(tokens=torch.zeros(1, 1), page_table=torch.zeros(1, 1))

    assert [event[0] for event in identity_events] == ["prepare", "invoke", "capture_plan"]
    assert all(event[1] is prepared for event in identity_events)
    assert len(registered) == 1


def test_traced_prefill_recompile_reuses_existing_trace_association(monkeypatch):
    prepared = _prepared_prefill()

    def prepare(
        *,
        tokens,
        page_table,
        prompt_lens=None,
        start_pos=None,
        empty_slots=None,
        sampling_params=None,
    ):
        return (prepared,)

    prefill = _runtime(
        PrefillRuntime,
        prepare=prepare,
        invoke=lambda prepared: PrefillInvocationResult(torch.zeros(1), ()),
        capture_plan=lambda prepared: (_ for _ in ()).throw(AssertionError("capture plan rebuilt")),
    )
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(prefill=prefill, decode=_runtime(DecodeRuntime), program_compiler=program_compiler)
    trace_compiler = _trace_compiler(program_compiler)
    trace_compiler.trace_key_for_program = lambda program_key: "existing-trace"
    trace_compiler.register_capture_plan = lambda plan: (_ for _ in ()).throw(AssertionError("plan registered"))
    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler)

    traced.compile_prefill(tokens=torch.zeros(1, 1), page_table=torch.zeros(1, 1))


def test_traced_decode_recompile_reuses_existing_trace_association(monkeypatch):
    prepared = _prepared_decode()

    def prepare(*, tokens, start_pos, page_table, sampling_params=None, reset_batch=False):
        return prepared

    decode = _runtime(
        DecodeRuntime,
        prepare=prepare,
        program_signature=lambda prepared: _Signature("decode", 1),
        invoke=lambda prepared, *, device_feedback=False: DecodeInvocationResult(torch.zeros(1), (), False),
        capture_plan=lambda prepared: (_ for _ in ()).throw(AssertionError("capture plan rebuilt")),
    )
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(prefill=_runtime(PrefillRuntime), decode=decode, program_compiler=program_compiler)
    trace_compiler = _trace_compiler(program_compiler)
    trace_compiler.trace_key_for_program = lambda program_key: "existing-trace"
    trace_compiler.register_capture_plan = lambda plan: (_ for _ in ()).throw(AssertionError("plan registered"))
    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler)

    traced.compile_decode(tokens=torch.zeros(1), start_pos=torch.zeros(1), page_table=torch.zeros(1, 1))


def test_execution_target_selection_is_external_to_traced_prefill(monkeypatch):
    prepared = _prepared_prefill(trace_eligible=True)
    invocations = []

    def prepare(
        *,
        tokens,
        page_table,
        prompt_lens=None,
        start_pos=None,
        empty_slots=None,
        sampling_params=None,
    ):
        return (prepared,)

    prefill = _runtime(
        PrefillRuntime,
        prepare=prepare,
        invoke=lambda prepared: invocations.append(prepared) or PrefillInvocationResult("eager", ()),
        assemble=lambda prepared_results, *, batch_size, sampling_params=None: prepared_results[0][1].value,
    )
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(prefill=prefill, decode=_runtime(DecodeRuntime), program_compiler=program_compiler)
    trace_compiler = _trace_compiler(program_compiler, mode="decode_only")

    def replay(
        program_key,
        refresh_inputs,
        *,
        reset_batch=False,
        device_feedback_enabled=False,
        feedback_compatible=False,
        page_table_changed=False,
    ):
        raise AssertionError("trace replayed")

    trace_compiler.replay = replay
    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler)

    assert (
        traced.eager_executor.prefill_forward(
            tokens=torch.zeros(1, 1),
            page_table=torch.zeros(1, 1),
        )
        == "eager"
    )
    assert invocations == [prepared]


def test_prefill_replay_does_not_interpret_request_eligibility(monkeypatch):
    prepared = _prepared_prefill(trace_eligible=False)
    prepared.trace_signature = _Signature("prefill-trace", 1)
    persistent = object()
    hidden = object()
    identity_events = []

    def prepare(
        *,
        tokens,
        page_table,
        prompt_lens=None,
        start_pos=None,
        empty_slots=None,
        sampling_params=None,
    ):
        identity_events.append(("prepare", prepared))
        return (prepared,)

    prefill = _runtime(
        PrefillRuntime,
        prepare=prepare,
        refresh_trace=lambda prepared, persistent: identity_events.append(("refresh", prepared, persistent)),
        finish_trace=lambda prepared, hidden, persistent: identity_events.append(
            ("finish", prepared, hidden, persistent)
        )
        or "traced",
        assemble=lambda prepared_results, *, batch_size, sampling_params=None: next(iter(prepared_results))[1],
    )
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(prefill=prefill, decode=_runtime(DecodeRuntime), program_compiler=program_compiler)
    trace_compiler = _trace_compiler(program_compiler)
    artifact = SimpleNamespace(persistent_inputs=SimpleNamespace(values=persistent))
    record = SimpleNamespace(artifact=artifact)
    trace_compiler.replay = (
        lambda program_key, refresh_inputs, *, reset_batch=False, device_feedback_enabled=False, feedback_compatible=False, page_table_changed=False: refresh_inputs(
            artifact, object()
        )
        or hidden
    )
    trace_compiler.trace_key_for_program = lambda program_key: "trace-key"
    trace_compiler.get = lambda trace_key: record
    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler)

    result = traced.prefill_forward(tokens=torch.zeros(1, 1), page_table=torch.zeros(1, 1))

    assert result == "traced"
    assert [event[0] for event in identity_events] == ["prepare", "refresh", "finish"]
    assert all(event[1] is prepared for event in identity_events)
    assert identity_events[1][2] is persistent
    assert identity_events[2][2:] == (hidden, persistent)


def test_prefill_replay_is_consumed_before_shared_trace_output_is_overwritten(monkeypatch):
    prepared = (
        _prepared_prefill(name="first"),
        _prepared_prefill(name="second"),
    )
    persistent = {"output": None}

    def prepare(**kwargs):
        return prepared

    def refresh_trace(request, trace_inputs):
        trace_inputs["output"] = request.name

    def assemble(prepared_results, *, batch_size, sampling_params=None):
        return [result.value["output"] for _, result in prepared_results]

    prefill = _runtime(
        PrefillRuntime,
        prepare=prepare,
        refresh_trace=refresh_trace,
        finish_trace=lambda request, hidden, trace_inputs: PrefillInvocationResult(trace_inputs, ()),
        assemble=assemble,
    )
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(prefill=prefill, decode=_runtime(DecodeRuntime), program_compiler=program_compiler)
    trace_compiler = _trace_compiler(program_compiler)
    artifact = SimpleNamespace(persistent_inputs=SimpleNamespace(values=persistent))
    trace_compiler.replay = lambda program_key, refresh_inputs, **kwargs: refresh_inputs(artifact, object()) or "hidden"
    trace_compiler.trace_key_for_program = lambda program_key: "trace-key"
    trace_compiler.get = lambda trace_key: SimpleNamespace(artifact=artifact)

    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler)

    assert traced.prefill_forward(tokens=torch.zeros(2, 1), page_table=torch.zeros(2, 1)) == [
        "first",
        "second",
    ]


def test_prefill_missing_trace_artifact_is_an_error_without_eager_reinvocation(monkeypatch, expect_error):
    prepared = _prepared_prefill(trace_eligible=True)
    eager_invocations = []

    def prepare(
        *,
        tokens,
        page_table,
        prompt_lens=None,
        start_pos=None,
        empty_slots=None,
        sampling_params=None,
    ):
        return (prepared,)

    prefill = _runtime(
        PrefillRuntime,
        prepare=prepare,
        invoke=lambda prepared: eager_invocations.append(prepared) or PrefillInvocationResult("eager", ()),
        refresh_trace=lambda prepared, persistent: None,
        assemble=lambda prepared_results, *, batch_size, sampling_params=None: next(iter(prepared_results))[1],
    )
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(prefill=prefill, decode=_runtime(DecodeRuntime), program_compiler=program_compiler)
    trace_compiler = _trace_compiler(program_compiler)
    artifact = SimpleNamespace(persistent_inputs=SimpleNamespace(values=()))
    trace_compiler.replay = (
        lambda program_key, refresh_inputs, *, reset_batch=False, device_feedback_enabled=False, feedback_compatible=False, page_table_changed=False: refresh_inputs(
            artifact, object()
        )
        or "hidden"
    )
    trace_compiler.trace_key_for_program = lambda program_key: "missing"
    trace_compiler.get = lambda trace_key: None
    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler)

    with expect_error(RuntimeError, "Required prefill trace") as exc_info:
        traced.prefill_forward(tokens=torch.zeros(1, 1), page_table=torch.zeros(1, 1))

    assert eager_invocations == []
    message = str(exc_info.value)
    for field in (
        "operation=prefill",
        "trace_mode=all",
        "model=PrefillRuntime",
        "signature_material=",
        "signature_digest=",
        "configured_coverage=",
        "TraceConfig(mode='none')",
    ):
        assert field in message
    assert traced.coverage_miss_count == 1


def test_prefill_missing_trace_signature_is_rejected_before_replay(monkeypatch, expect_error):
    prepared = _prepared_prefill(trace_eligible=False)
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(
        prefill=_runtime(PrefillRuntime),
        decode=_runtime(DecodeRuntime),
        program_compiler=program_compiler,
    )
    trace_compiler = _trace_compiler(program_compiler)
    trace_compiler.replay = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected replay"))
    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler)

    with expect_error(RuntimeError, "not trace-eligible"):
        traced._execute_prefill(prepared)


def test_decode_missing_trace_artifact_reports_exact_strict_coverage(monkeypatch, expect_error):
    prepared = _prepared_decode()
    signature = _Signature("decode", 7)
    decode = _runtime(
        DecodeRuntime,
        config=SimpleNamespace(position_feedback_capable=True),
        program_signature=lambda value: signature,
    )
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(prefill=_runtime(PrefillRuntime), decode=decode, program_compiler=program_compiler)
    trace_compiler = _trace_compiler(program_compiler)
    trace_compiler.trace_key_for_program = lambda key: None
    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler, trace_mode="decode_only")

    with expect_error(RuntimeError, "Required decode trace") as exc_info:
        traced._execute_decode(prepared)

    message = str(exc_info.value)
    for field in (
        "operation=decode",
        "trace_mode=decode_only",
        "model=DecodeRuntime",
        "signature_material=",
        "signature_digest=",
        "program_key=",
        "trace_key=unavailable",
        "configured_coverage=",
        "TraceConfig(mode='none')",
    ):
        assert field in message
    assert traced.coverage_miss_count == 1


def test_prefill_whole_call_preflights_every_trace_before_first_replay(monkeypatch, expect_error):
    first = _prepared_prefill(signatures=(_Signature("prefill", 1),), name="first")
    second = _prepared_prefill(signatures=(_Signature("prefill", 2),), name="second")
    prefill = _runtime(
        PrefillRuntime,
        prepare=lambda **kwargs: (first, second),
        assemble=lambda results, **kwargs: tuple(results),
    )
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(prefill=prefill, decode=_runtime(DecodeRuntime), program_compiler=program_compiler)
    trace_compiler = _trace_compiler(program_compiler)
    first_key = program_compiler.key_for(first.program_signatures[0])
    second_key = program_compiler.key_for(second.program_signatures[0])
    artifact = SimpleNamespace(persistent_inputs=SimpleNamespace(values=()))
    trace_compiler.trace_key_for_program = lambda key: "first" if key == first_key else "second"
    trace_compiler.get = lambda key: SimpleNamespace(artifact=artifact) if key == "first" else None
    replays = []
    trace_compiler.replay = lambda *args, **kwargs: replays.append(args) or "hidden"
    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler)

    with expect_error(RuntimeError, second_key.digest):
        traced.prefill_forward(tokens=torch.zeros(2, 1), page_table=torch.zeros(2, 1))

    assert replays == []


def test_prefill_fixed_chunk_replays_every_step_and_finishes_only_final_hidden(monkeypatch):
    prepared = _prepared_prefill()
    chunks = ("chunk-0", "chunk-1", "chunk-2")
    prepared.request = SimpleNamespace(chunks=chunks)
    events = []
    prefill = _runtime(
        PrefillRuntime,
        refresh_trace=lambda request, hidden_inputs, workspace, chunk: events.append(("refresh", chunk, workspace)),
        finish_trace=lambda request, hidden, persistent: events.append(("finish", hidden, persistent))
        or PrefillInvocationResult(hidden, ()),
    )
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(prefill=prefill, decode=_runtime(DecodeRuntime), program_compiler=program_compiler)
    trace_compiler = _trace_compiler(program_compiler)
    persistent = object()
    artifact = SimpleNamespace(persistent_inputs=SimpleNamespace(values=persistent))
    record = SimpleNamespace(artifact=artifact)
    trace_compiler.trace_key_for_program = lambda key: "trace"
    trace_compiler.get = lambda key: record
    trace_compiler.workspace_for_program = lambda key: persistent
    replayed = []

    def replay(program_key, refresh_inputs, **kwargs):
        refresh_inputs(artifact, object())
        replayed.append(program_key)
        return f"hidden-{len(replayed) - 1}"

    trace_compiler.replay = replay
    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler)

    result = traced._execute_prefill(prepared)

    assert result.value == "hidden-2"
    assert [event[:2] for event in events] == [
        ("refresh", "chunk-0"),
        ("refresh", "chunk-1"),
        ("refresh", "chunk-2"),
        ("finish", "hidden-2"),
    ]
    assert all(event[-1] is persistent for event in events)


def test_prefill_replay_emits_structured_serving_evidence(monkeypatch):
    class Request:
        source_rows = tuple(range(15))
        padded_batch_size = 16
        padded_sequence_length = 128
        chunks = ("step",)

    monkeypatch.setattr(execution_module, "PrefillRequest", Request)
    signature = SimpleNamespace(
        operation_variant="regular-batched",
        key_material=(("operation_variant", "regular-batched"),),
    )
    prepared = SimpleNamespace(
        request=Request(),
        program_signatures=(signature,),
        trace_signature=SimpleNamespace(key_material=(("padded_batch_size", 16),)),
        sampling_path="topk",
        sampling_params=object(),
    )
    persistent = object()
    prefill = _runtime(
        PrefillRuntime,
        refresh_trace=lambda request, hidden_inputs, workspace, chunk: None,
        finish_trace=lambda request, hidden, workspace: PrefillInvocationResult(hidden, ()),
        assemble=lambda results, **kwargs: next(iter(results))[1],
    )
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(prefill=prefill, decode=_runtime(DecodeRuntime), program_compiler=program_compiler)
    trace_compiler = _trace_compiler(program_compiler)
    program_key = program_compiler.key_for(signature)
    trace_key = SimpleNamespace(digest="1" * 64)
    artifact = SimpleNamespace(persistent_inputs=SimpleNamespace(values=persistent))
    record = SimpleNamespace(artifact=artifact)
    trace_compiler.trace_key_for_program = lambda key: trace_key
    trace_compiler.workspace_for_program = lambda key: persistent
    trace_compiler.replay = lambda key, refresh, **kwargs: refresh(artifact, object()) or "hidden"
    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler)

    result = traced.execute_prepared_prefill(
        ((prepared, ((program_key, record),)),),
        batch_size=15,
        sampling_params=prepared.sampling_params,
        lane=3,
    )

    assert result.value == "hidden"
    assert len(traced.recent_prefill_replay_evidence) == 1
    evidence = traced.recent_prefill_replay_evidence[0]
    assert evidence.operation == "prefill"
    assert evidence.variant == "regular-batched"
    assert evidence.sampling_path == "topk"
    assert evidence.execution == "trace_replay"
    assert (evidence.active_batch_size, evidence.padded_batch_size) == (15, 16)
    assert evidence.padded_sequence_length == 128
    assert (evidence.lane, evidence.rank) == (3, 3)
    assert evidence.program_key == program_key.digest
    assert evidence.trace_key == "1" * 64
    assert evidence.replay_steps == 1
    assert traced.runtime_summary() == {
        "eager_prefill_executions": 0,
        "semantic_program_count": 0,
        "rejected_post_activation_compile_attempts": 0,
        "ttnn_program_cache_count": None,
        "successful_trace_replays": 0,
        "trace_replays_by_operation": {"prefill": 0, "decode": 0},
        "strict_coverage_misses": 0,
        "semantic_trace_count": 0,
        "trace_association_count": 0,
    }


def test_decode_replay_prepares_once_and_uses_same_object_for_refresh_submission_and_consume(monkeypatch):
    prepared = _prepared_decode()
    events = []

    def prepare(*, tokens, start_pos, page_table, sampling_params=None, reset_batch=False):
        events.append(("prepare", prepared, sampling_params, reset_batch))
        return prepared

    decode = _runtime(
        DecodeRuntime,
        config=SimpleNamespace(position_feedback_capable=True),
        prepare=prepare,
        program_signature=lambda prepared: events.append(("signature", prepared)) or _Signature("decode", 1),
        refresh_trace=lambda artifact, prepared, decision: events.append(("refresh", prepared)),
        note_submitted=lambda prepared: events.append(("submitted", prepared)),
        consume=lambda result, *, read_from_device=True: events.append(("consume", result, read_from_device))
        or result.value,
    )
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(prefill=_runtime(PrefillRuntime), decode=decode, program_compiler=program_compiler)
    trace_compiler = _trace_compiler(program_compiler)
    trace_key = SimpleNamespace(digest="2" * 64)
    trace_compiler.trace_key_for_program = lambda key: trace_key
    trace_compiler.get = lambda key: SimpleNamespace(artifact=object())
    trace_compiler.replay = (
        lambda program_key, refresh_inputs, *, reset_batch=False, device_feedback_enabled=False, feedback_compatible=False, page_table_changed=False: refresh_inputs(
            object(), object()
        )
        or "token"
    )
    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler)
    sampling_params = object()

    result = traced.decode_forward(
        tokens=torch.zeros(1, 1),
        start_pos=torch.zeros(1),
        page_table=torch.zeros(1, 1),
        sampling_params=sampling_params,
        reset_batch=True,
        read_from_device=False,
    )

    assert result == "token"
    assert [event[0] for event in events] == ["prepare", "signature", "refresh", "submitted", "consume"]
    assert all(event[1] is prepared for event in events[:-1])
    assert events[0][2:] == (sampling_params, True)
    assert isinstance(events[-1][1], DecodeInvocationResult)
    assert events[-1][1].owned is None
    assert events[-1][2] is False


def test_explicit_eager_decode_delegates_once_and_execution_objects_do_not_cleanup(monkeypatch):
    prepared = _prepared_decode()
    calls = []

    def prepare(*, tokens, start_pos, page_table, sampling_params=None, reset_batch=False):
        calls.append(("prepare", prepared, sampling_params, reset_batch))
        return prepared

    decode = _runtime(
        DecodeRuntime,
        prepare=prepare,
        invoke=lambda prepared, *, device_feedback=False: calls.append(("invoke", prepared, device_feedback))
        or DecodeInvocationResult("eager", (), False),
        consume=lambda result, *, read_from_device=True: calls.append(("consume", result.value, read_from_device))
        or result.value,
    )
    program_compiler = _compiler(monkeypatch)
    eager = EagerExecutor(prefill=_runtime(PrefillRuntime), decode=decode, program_compiler=program_compiler)
    trace_compiler = _trace_compiler(program_compiler)

    def replay(
        program_key,
        refresh_inputs,
        *,
        reset_batch=False,
        device_feedback_enabled=False,
        feedback_compatible=False,
        page_table_changed=False,
    ):
        raise AssertionError("trace replayed")

    trace_compiler.replay = replay
    traced = TracedExecutor(eager=eager, trace_compiler=trace_compiler)
    sampling_params = object()

    assert (
        traced.eager_executor.decode_forward(
            tokens=torch.zeros(1, 1),
            start_pos=torch.zeros(1),
            page_table=torch.zeros(1, 1),
            sampling_params=sampling_params,
            reset_batch=True,
            read_from_device=False,
        )
        == "eager"
    )
    assert calls == [
        ("prepare", prepared, sampling_params, True),
        ("invoke", prepared, False),
        ("consume", "eager", False),
    ]

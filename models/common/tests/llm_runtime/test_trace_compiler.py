# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace

import torch

import models.common.llm_runtime.trace_compiler as trace_compiler_module
import ttnn
from models.common.llm_runtime.decode import DecodeDeviceInputs, DecodePersistentInputs
from models.common.llm_runtime.prefill import PrefillDeviceInputs, PrefillPersistentInputs, PrefillPositionInputs
from models.common.llm_runtime.program_compiler import ProgramCompiler
from models.common.llm_runtime.trace_compiler import InputRefreshPolicy, TraceCapturePlan, TraceCompiler


@dataclass(frozen=True)
class _Signature:
    kind: str
    variant: int

    @property
    def key_material(self):
        return (("kind", self.kind), ("variant", self.variant))


def _patch_backend(monkeypatch, events):
    next_trace_id = iter(range(100, 200))
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: events.append(("sync", mesh)))
    monkeypatch.setattr(
        ttnn,
        "begin_trace_capture",
        lambda mesh, cq_id: events.append(("begin", mesh, cq_id)) or next(next_trace_id),
    )
    monkeypatch.setattr(
        ttnn,
        "end_trace_capture",
        lambda mesh, trace_id, cq_id: events.append(("end", trace_id, cq_id)),
    )
    monkeypatch.setattr(
        ttnn,
        "execute_trace",
        lambda mesh, trace_id, cq_id, blocking: events.append(("execute", trace_id, cq_id, blocking)),
    )
    monkeypatch.setattr(ttnn, "release_trace", lambda mesh, trace_id: events.append(("release", trace_id)))
    monkeypatch.setattr(trace_compiler_module, "trim_host_allocator", lambda: events.append(("trim",)))


def _compiled_program(program_compiler, monkeypatch, variant):
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: None)
    return program_compiler.compile(_Signature("program", variant), lambda context: torch.zeros(1))


def _plan(program, variant, events, *, operation="decode", policy=InputRefreshPolicy()):
    return TraceCapturePlan(
        program_key=program.key,
        trace_signature=_Signature("trace", variant),
        operation=operation,
        prepare_inputs=lambda: events.append(("prepare", variant)) or (),
        capture=lambda persistent: events.append(("capture", variant)) or torch.zeros(1),
        refresh_policy=policy,
    )


def test_trace_compiler_retains_exact_program_compiler_and_separate_registries(monkeypatch):
    compiler = ProgramCompiler("mesh", lambda: object())
    program = _compiled_program(compiler, monkeypatch, 1)
    trace = TraceCompiler(compiler, "mesh", SimpleNamespace(mode="all"))
    trace_key = trace.register_capture_plan(_plan(program, 1, []))

    assert trace.program_compiler is compiler
    assert trace.program_to_trace == {program.key: trace_key}
    assert set(trace.traces) == {trace_key}
    assert compiler.programs == {program.key: program}
    assert not hasattr(program, "artifact")
    assert not hasattr(trace, "programs")


def test_trace_aliases_share_one_artifact_without_copying_program_records(monkeypatch):
    events = []
    _patch_backend(monkeypatch, events)
    compiler = ProgramCompiler("mesh", lambda: object())
    first = compiler.compile(_Signature("program", 1), lambda context: torch.zeros(1))
    second = compiler.compile(_Signature("program", 2), lambda context: torch.zeros(1))
    trace = TraceCompiler(compiler, "mesh", SimpleNamespace(mode="all"))
    shared_signature = _Signature("trace", 1)
    first_key = trace.register_capture_plan(
        TraceCapturePlan(first.key, shared_signature, "decode", lambda: (), lambda persistent: torch.zeros(1))
    )
    second_key = trace.register_capture_plan(
        TraceCapturePlan(second.key, shared_signature, "decode", lambda: (), lambda persistent: torch.zeros(1))
    )

    trace.capture_all()

    assert first_key == second_key
    assert len(trace.traces) == 1
    assert trace.program_to_trace == {first.key: first_key, second.key: first_key}
    assert len(compiler.programs) == 2
    assert [event[0] for event in events].count("begin") == 1


def test_capture_allocates_every_input_before_capture_and_coordinates_gates(monkeypatch, expect_error):
    events = []
    _patch_backend(monkeypatch, events)
    compiler = ProgramCompiler("mesh", lambda: object())
    programs = [compiler.compile(_Signature("program", variant), lambda context: torch.zeros(1)) for variant in (1, 2)]
    trace = TraceCompiler(compiler, "mesh", SimpleNamespace(mode="all"))

    def capture_with_gate(persistent):
        events.append(("capture", 1))
        with expect_error(RuntimeError, "capture is in progress"):
            compiler.compile(_Signature("program", 3), lambda context: torch.zeros(1))
        return torch.zeros(1)

    trace.register_capture_plan(
        TraceCapturePlan(
            programs[0].key,
            _Signature("trace", 1),
            "decode",
            lambda: events.append(("prepare", 1)) or (),
            capture_with_gate,
        )
    )
    trace.register_capture_plan(_plan(programs[1], 2, events))
    events.clear()

    trace.capture_all()

    first_begin = next(index for index, event in enumerate(events) if event[0] == "begin")
    assert events[:first_begin] == [("prepare", 1), ("prepare", 2)]
    assert trace.trace_active and compiler.trace_active
    assert not trace.capture_in_progress and not compiler.trace_capture_in_progress
    with expect_error(RuntimeError, "after trace activation"):
        compiler.compile(_Signature("program", 3), lambda context: torch.zeros(1))


def test_capture_orders_decode_before_prefill_after_allocating_every_input(monkeypatch):
    events = []
    _patch_backend(monkeypatch, events)
    compiler = ProgramCompiler("mesh", lambda: object())
    programs = [compiler.compile(_Signature("program", variant), lambda context: torch.zeros(1)) for variant in (1, 2)]
    trace = TraceCompiler(compiler, "mesh", SimpleNamespace(mode="all"))
    trace.register_capture_plan(_plan(programs[0], 1, events, operation="prefill"))
    trace.register_capture_plan(_plan(programs[1], 2, events, operation="decode"))
    events.clear()

    trace.capture_all()

    first_begin = next(index for index, event in enumerate(events) if event[0] == "begin")
    assert events[:first_begin] == [("prepare", 1), ("prepare", 2)]
    assert [event for event in events if event[0] == "capture"] == [("capture", 2), ("capture", 1)]


def test_trace_mode_skips_disabled_or_ineligible_associations(monkeypatch):
    compiler = ProgramCompiler("mesh", lambda: object())
    prefill = _compiled_program(compiler, monkeypatch, 1)
    decode = _compiled_program(compiler, monkeypatch, 2)
    trace = TraceCompiler(compiler, "mesh", SimpleNamespace(mode="decode_only"))

    assert trace.register_capture_plan(_plan(prefill, 1, [], operation="prefill")) is None
    assert (
        trace.register_capture_plan(
            TraceCapturePlan(
                decode.key,
                _Signature("trace", 2),
                "decode",
                lambda: (),
                lambda persistent: torch.zeros(1),
                trace_eligible=False,
            )
        )
        is None
    )
    assert trace.program_to_trace == {}
    assert trace.traces == {}


def test_capture_failure_rolls_back_traces_and_uncaptured_inputs(monkeypatch, expect_error):
    events = []
    _patch_backend(monkeypatch, events)

    class OwnedTensor:
        pass

    released = []
    monkeypatch.setattr(ttnn, "Tensor", OwnedTensor)
    monkeypatch.setattr(ttnn, "deallocate", released.append)
    compiler = ProgramCompiler("mesh", lambda: object())
    programs = [compiler.compile(_Signature("program", variant), lambda context: torch.zeros(1)) for variant in (1, 2)]
    trace = TraceCompiler(compiler, "mesh", SimpleNamespace(mode="all"))
    first_input, first_output, second_input = OwnedTensor(), OwnedTensor(), OwnedTensor()
    trace.register_capture_plan(
        TraceCapturePlan(
            programs[0].key,
            _Signature("trace", 1),
            "decode",
            lambda: first_input,
            lambda persistent: first_output,
        )
    )
    primary = RuntimeError("second capture failed")
    trace.register_capture_plan(
        TraceCapturePlan(
            programs[1].key,
            _Signature("trace", 2),
            "decode",
            lambda: second_input,
            lambda persistent: (_ for _ in ()).throw(primary),
        )
    )

    with expect_error(RuntimeError, "second capture failed") as caught:
        trace.capture_all()

    assert caught.value is primary
    assert released.count(first_input) == 1
    assert released.count(first_output) == 1
    assert released.count(second_input) == 1
    assert not trace.trace_active and not compiler.trace_active
    assert all(record.artifact is None for record in trace.traces.values())


def test_incomplete_capture_rollback_keeps_program_gate_closed_until_cleanup(monkeypatch, expect_error):
    events = []
    _patch_backend(monkeypatch, events)

    class OwnedTensor:
        pass

    retry = OwnedTensor()
    attempts = []

    def deallocate(value):
        attempts.append(value)
        if value is retry and attempts.count(retry) == 1:
            raise RuntimeError("release once")

    monkeypatch.setattr(ttnn, "Tensor", OwnedTensor)
    monkeypatch.setattr(ttnn, "deallocate", deallocate)
    compiler = ProgramCompiler("mesh", lambda: object())
    programs = [compiler.compile(_Signature("program", variant), lambda context: torch.zeros(1)) for variant in (1, 2)]
    trace = TraceCompiler(compiler, "mesh", SimpleNamespace(mode="all"))
    trace.register_capture_plan(
        TraceCapturePlan(programs[0].key, _Signature("trace", 1), "decode", lambda: retry, lambda _: torch.zeros(1))
    )
    trace.register_capture_plan(
        TraceCapturePlan(
            programs[1].key,
            _Signature("trace", 2),
            "decode",
            lambda: (_ for _ in ()).throw(RuntimeError("prepare failed")),
            lambda _: torch.zeros(1),
        )
    )

    with expect_error(RuntimeError, "prepare failed"):
        trace.capture_all()

    assert trace.trace_active and compiler.trace_active
    assert trace.rollback_orphan_count == 1
    with expect_error(RuntimeError, "after trace activation"):
        compiler.compile(_Signature("program", 3), lambda context: torch.zeros(1))

    trace.cleanup()
    assert attempts.count(retry) == 2
    assert not compiler.trace_active


def test_replay_refresh_decisions_cover_first_replay_page_change_feedback_and_switch(monkeypatch):
    events = []
    _patch_backend(monkeypatch, events)
    compiler = ProgramCompiler("mesh", lambda: object())
    programs = [compiler.compile(_Signature("program", variant), lambda context: torch.zeros(1)) for variant in (1, 2)]
    policy = InputRefreshPolicy(every_replay=("position", "sampling"))
    trace = TraceCompiler(compiler, "mesh", SimpleNamespace(mode="all"))
    for variant, program in enumerate(programs, 1):
        trace.register_capture_plan(_plan(program, variant, events, policy=policy))
    trace.capture_all()
    decisions = []

    trace.replay(
        programs[0].key,
        lambda artifact, decision: decisions.append(decision),
        device_feedback_enabled=True,
        feedback_compatible=True,
    )
    trace.replay(
        programs[0].key,
        lambda artifact, decision: decisions.append(decision),
        device_feedback_enabled=True,
        feedback_compatible=True,
        page_table_changed=True,
    )
    trace.replay(
        programs[1].key,
        lambda artifact, decision: decisions.append(decision),
        device_feedback_enabled=True,
        feedback_compatible=True,
    )
    trace.replay(
        programs[1].key,
        lambda artifact, decision: decisions.append(decision),
        device_feedback_enabled=False,
    )

    assert [decision.full for decision in decisions] == [True, False, True, True]
    assert [decision.page_table for decision in decisions] == [False, True, False, False]
    assert all(decision.fields == ("position", "sampling") for decision in decisions)
    assert [event[0] for event in events].count("execute") == 4


def test_cleanup_retries_trace_release_before_deallocating_and_does_not_own_programs(monkeypatch, expect_error):
    events = []
    _patch_backend(monkeypatch, events)

    class OwnedTensor:
        pass

    persistent, output = OwnedTensor(), OwnedTensor()
    deallocated = []
    release_attempts = []

    def release(mesh, trace_id):
        release_attempts.append(trace_id)
        if len(release_attempts) == 1:
            raise RuntimeError("trace release once")

    monkeypatch.setattr(ttnn, "Tensor", OwnedTensor)
    monkeypatch.setattr(ttnn, "deallocate", deallocated.append)
    monkeypatch.setattr(ttnn, "release_trace", release)
    compiler = ProgramCompiler("mesh", lambda: object())
    program = compiler.compile(_Signature("program", 1), lambda context: torch.zeros(1))
    trace = TraceCompiler(compiler, "mesh", SimpleNamespace(mode="all"))
    trace.register_capture_plan(
        TraceCapturePlan(
            program.key,
            _Signature("trace", 1),
            "decode",
            lambda: persistent,
            lambda values: output,
        )
    )
    trace.capture_all()

    with expect_error(RuntimeError, "Failed to release"):
        trace.cleanup()
    assert deallocated == []
    assert program.ready

    trace.cleanup()
    trace.cleanup()
    assert release_attempts == [100, 100]
    assert deallocated.count(persistent) == 1
    assert deallocated.count(output) == 1
    assert program.ready

    compiler.cleanup()
    assert not program.ready


def test_cleanup_releases_operation_owned_persistent_dataclasses_once(monkeypatch):
    events = []
    _patch_backend(monkeypatch, events)

    class OwnedTensor:
        pass

    values = [OwnedTensor() for _ in range(19)]
    decode = DecodePersistentInputs(
        DecodeDeviceInputs(*values[:4]),
        tuple(values[4:7]),
    )
    prefill = PrefillPersistentInputs(
        PrefillDeviceInputs(*values[7:14]),
        PrefillPositionInputs(*values[14:17]),
        (values[17], values[17], values[17]),
        values[18],
    )
    deallocated = []
    monkeypatch.setattr(ttnn, "Tensor", OwnedTensor)
    monkeypatch.setattr(ttnn, "deallocate", deallocated.append)

    compiler = ProgramCompiler("mesh", lambda: object())
    programs = [compiler.compile(_Signature("program", variant), lambda context: torch.zeros(1)) for variant in (1, 2)]
    trace = TraceCompiler(compiler, "mesh", SimpleNamespace(mode="all"))
    trace.register_capture_plan(
        TraceCapturePlan(programs[0].key, _Signature("trace", 1), "decode", lambda: decode, lambda _: values[0])
    )
    trace.register_capture_plan(
        TraceCapturePlan(programs[1].key, _Signature("trace", 2), "prefill", lambda: prefill, lambda _: values[18])
    )

    trace.capture_all()
    trace.cleanup()

    assert len(deallocated) == len(values)
    assert {id(value) for value in deallocated} == {id(value) for value in values}

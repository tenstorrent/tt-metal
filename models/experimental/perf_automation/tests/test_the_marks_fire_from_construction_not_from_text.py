"""The AST injector places a call in text; this fires from the model's own objects instead.

Voxtral's marked pass needed the test's real batch bound onto the pipeline BEFORE the marks ran, or
its own <stage>_trace_inputs hooks would raise FileNotFoundError on the tensors this tree does not
ship (agent/captured_stub.py's own docstring). A hook that fires at construction -- before the test's
preparer has bound anything -- is too early for that model. nemotron has no preparer at all and marks
correctly the instant its pipeline exists. So runtime_marks.install() must wrap the test's preparer
when one exists, and fall back to wrapping construction only when it does not.
"""
from __future__ import annotations

import sys
import types

import pytest

from agent import runtime_marks, stage_marks as sm


@pytest.fixture(autouse=True)
def _no_real_marking(monkeypatch):
    """Every test here checks WHAT install() wires up and WHEN it fires -- never real hardware."""
    calls = []
    monkeypatch.setattr(sm, "mark_stages_for", lambda pipe, device: calls.append((pipe, device)) or 1)
    return calls


class _Pipe:
    PIPELINE_STAGES = ["encode", "decode"]

    def __init__(self, device):
        self.device = device


@pytest.fixture
def _fake_module(monkeypatch):
    """A throwaway module, source-backed so inspect.getsource (which find_input_preparer needs) and
    find_pipeline_classes both see real text/attributes instead of a live object's introspection."""
    created = []

    def _make(name: str, source: str = "", **attrs):
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
        created.append(name)
        real_getsource = __import__("inspect").getsource
        monkeypatch.setattr(
            "inspect.getsource", lambda obj, _m=mod, _s=source, _r=real_getsource: (_s if obj is _m else _r(obj))
        )
        return mod

    yield _make
    for name in created:
        sys.modules.pop(name, None)


def test_a_module_level_preparer_is_wrapped_and_fired_after_it_runs(_no_real_marking, _fake_module):
    """The voxtral shape: `_patch_trace_inputs(pipe, batch)` binds the real batch onto the pipeline's
    hooks. The marked pass must run AFTER that call returns, using the SAME pipe -- not before, which
    is what construction-time firing would have done."""
    order = []
    src = "def _patch_trace_inputs(pipe, batch):\n    order.append('preparer')\n    pipe.decode_trace_inputs = lambda: batch\n"
    ns = {"order": order}
    exec(compile(src, "<fake voxtral test>", "exec"), ns)  # noqa: S102 -- test fixture text, not model code
    mod = _fake_module("tt_voxtral_test_mod", source=src, _patch_trace_inputs=ns["_patch_trace_inputs"])

    restore = runtime_marks.install(mod)
    try:
        pipe = _Pipe(device="the-device")
        mod._patch_trace_inputs(pipe, batch="the-real-batch")
    finally:
        restore()

    assert order == ["preparer"], "the marked pass must run after the preparer, not before it"
    assert _no_real_marking == [(pipe, "the-device")]
    assert pipe.decode_trace_inputs() == "the-real-batch", "the preparer's own binding must still run"


def test_a_test_with_no_preparer_fires_the_instant_a_pipeline_is_built(_no_real_marking, _fake_module):
    """The nemotron shape: no preparer anywhere in the test, so there is nothing to wait for -- the
    marked pass fires the moment a pipeline-shaped object is constructed."""
    mod = _fake_module("tt_nemotron_test_mod", source="", Pipe=_Pipe)

    restore = runtime_marks.install(mod)
    try:
        pipe = _Pipe(device="dev-2")
    finally:
        restore()

    assert _no_real_marking == [(pipe, "dev-2")]


def test_it_fires_at_most_once_even_with_two_pipelines(_no_real_marking, _fake_module):
    mod = _fake_module("tt_two_pipes_test_mod", source="", Pipe=_Pipe)

    restore = runtime_marks.install(mod)
    try:
        _Pipe(device="a")
        _Pipe(device="b")
    finally:
        restore()

    assert len(_no_real_marking) == 1, "a second pipeline construction must not mark again"


def test_restore_puts_back_the_original_init(_no_real_marking, _fake_module):
    mod = _fake_module("tt_restore_test_mod", source="", Pipe=_Pipe)
    original_init = _Pipe.__init__

    restore = runtime_marks.install(mod)
    restore()

    assert _Pipe.__init__ is original_init


def test_a_module_with_nothing_to_wrap_is_a_safe_no_op(_no_real_marking, _fake_module):
    mod = _fake_module("tt_empty_test_mod", source="")

    restore = runtime_marks.install(mod)
    restore()
    restore()  # idempotent: calling it twice must not raise

    assert _no_real_marking == []


def test_device_comes_off_the_pipe_attribute_not_a_local(_no_real_marking, _fake_module):
    """runtime_marks never has a `device` local to read, unlike the text-injected pass -- it must
    take it off the pipe object itself, the same attribute every real pipeline already sets."""
    mod = _fake_module("tt_device_test_mod", source="", Pipe=_Pipe)

    restore = runtime_marks.install(mod)
    try:
        built = _Pipe(device="the-real-device")
    finally:
        restore()

    assert _no_real_marking == [(built, "the-real-device")]


def test_no_device_at_all_says_why_instead_of_raising(_no_real_marking, _fake_module, capsys):
    class _NoDevice:
        PIPELINE_STAGES = ["decode"]

    mod = _fake_module("tt_no_device_test_mod", source="", Pipe=_NoDevice)

    restore = runtime_marks.install(mod)
    try:
        _NoDevice()
    finally:
        restore()

    err = capsys.readouterr().err
    assert "NO per-stage boundaries" in err and "no .device" in err
    assert _no_real_marking == [], "marking must not be attempted without a device"


def test_a_preparer_outside_module_scope_is_not_mistaken_for_one(_no_real_marking, _fake_module):
    """find_input_preparer only trusts module-level assignments (voxtral's and nemotron's real shape);
    a helper nested inside another function is invisible at module scope, matching the existing rule
    for what find_input_preparer accepts everywhere else in this file."""
    src = (
        "def _outer():\n"
        "    def _nested_preparer(pipe, batch):\n"
        "        pipe.decode_trace_inputs = lambda: batch\n"
        "    return _nested_preparer\n"
    )
    mod = _fake_module("tt_nested_prep_test_mod", source=src, Pipe=_Pipe)

    restore = runtime_marks.install(mod)
    try:
        pipe = _Pipe(device="dev-3")
    finally:
        restore()

    assert _no_real_marking == [(pipe, "dev-3")], "no module-level preparer found -> falls back to construction"

"""The separate per-stage pass is a switch; off, the same marks go on the measured forward itself.

WH Galaxy, 2026-09-28: the pass ran every declared stage a second time and cost 10,458 of the 32,769
ops tracy can record per chip, so the measured forward was cut off at 65%. With the pass off, each
stage's own method -- read from the model's <stage>_trace_step hook -- is wrapped on the pipeline
instance and emits its marks as the forward calls it. That is the default; the pass is opt-in
(PERF_MCP_STAGE_PASS=1) and remains the fallback for a pipeline whose stages cannot be matched.
Stage names here are the fake pipeline's own; the code under test never types one.
"""

import sys
import types

import pytest

from agent import stage_marks as sm

STAGES = ["alpha", "beta", "gamma"]


class _Pipe:
    PIPELINE_STAGES = STAGES

    def __init__(self):
        self.calls = []

    # the model's own stage methods, as its forward calls them
    def run_a(self, x):
        self.calls.append("a")
        return x + 1

    def run_b(self, x):
        self.calls.append("b")
        return x * 2

    def run_c(self, x):
        self.calls.append("c")
        if x < 0:
            raise ValueError("boom")
        return x

    def _prep(self):
        return 1

    # the seams: each calls a shared helper and exactly one method of its own
    def alpha_trace_step(self):
        return self.run_a(self._prep())

    def beta_trace_step(self):
        return self.run_b(self._prep())

    def gamma_trace_step(self):
        return self.run_c(self._prep())

    def forward(self, x):
        return self.run_c(self.run_b(self.run_a(x)))


@pytest.fixture
def marks(monkeypatch):
    seen = []
    monkeypatch.setattr(sm, "signpost", seen.append)
    return seen


def test_the_pass_is_off_unless_switched_on(monkeypatch):
    monkeypatch.delenv(sm.STAGE_PASS_ENV, raising=False)
    assert not sm.stage_pass_enabled()
    monkeypatch.setenv(sm.STAGE_PASS_ENV, "0")
    assert not sm.stage_pass_enabled()
    monkeypatch.setenv(sm.STAGE_PASS_ENV, "1")
    assert sm.stage_pass_enabled()


def test_each_stage_resolves_to_its_own_method():
    assert sm.stage_methods(_Pipe()) == {"alpha": "run_a", "beta": "run_b", "gamma": "run_c"}


def test_the_forward_carries_the_marks(marks):
    p = _Pipe()
    assert sm.mark_stages_on_forward(p) == 3
    assert p.forward(1) == 4
    assert marks == [
        "stage:alpha", "stage:alpha:end",
        "stage:beta", "stage:beta:end",
        "stage:gamma", "stage:gamma:end",
    ]  # fmt: skip
    assert p.calls == ["a", "b", "c"], "the stage methods still run, once each"


def test_a_stage_that_raises_still_closes_its_window(marks):
    p = _Pipe()
    sm.mark_stages_on_forward(p)
    with pytest.raises(ValueError):  # allow-pytest.raises: no expect_error fixture
        p.run_c(-1)
    assert marks == ["stage:gamma", "stage:gamma:end"]


def test_only_this_pipeline_is_marked(marks):
    p, other = _Pipe(), _Pipe()
    sm.mark_stages_on_forward(p)
    other.forward(1)
    assert marks == [], "a second pipeline (a trace-replay build) is untouched"


def test_an_ambiguous_stage_is_not_guessed():
    class _Shared(_Pipe):
        def beta_trace_step(self):
            return self.run_a(self._prep())  # the same method as alpha: no stage owns it

    assert sm.stage_methods(_Shared()) == {}


def test_a_stage_without_a_hook_is_not_guessed():
    class _Missing(_Pipe):
        PIPELINE_STAGES = STAGES + ["delta"]

    assert sm.stage_methods(_Missing()) == {}


def test_by_default_the_scope_is_marked_without_the_pass(monkeypatch, marks):
    monkeypatch.delenv(sm.STAGE_PASS_ENV, raising=False)
    ran = []
    monkeypatch.setattr(sm, "mark_stages_for", lambda pipe, device: ran.append(pipe) or 9)
    p = _Pipe()
    assert sm.mark_stages_in_scope({"pipe": p}) == 3
    assert ran == [], "no separate pass"
    p.forward(1)
    assert len(marks) == 6


def test_switched_off_an_unmatched_pipeline_falls_back_to_the_pass(monkeypatch):
    class _Shared(_Pipe):
        def beta_trace_step(self):
            return self.run_a(self._prep())

    monkeypatch.delenv(sm.STAGE_PASS_ENV, raising=False)
    ran = []
    monkeypatch.setattr(sm, "mark_stages_for", lambda pipe, device: ran.append(pipe) or 3)
    assert sm.mark_stages_in_scope({"pipe": _Shared()}) == 3
    assert len(ran) == 1


def test_switched_on_the_pass_runs_as_before(monkeypatch):
    monkeypatch.setenv(sm.STAGE_PASS_ENV, "1")
    ran = []
    monkeypatch.setattr(sm, "mark_stages_for", lambda pipe, device: ran.append(pipe) or 3)
    p = _Pipe()
    assert sm.mark_stages_in_scope({"pipe": p}) == 3
    assert ran == [p]
    assert p.run_a.__func__ is _Pipe.run_a, "nothing wrapped when the pass is on"


def test_a_module_level_stage_list_still_counts(monkeypatch):
    mod = types.ModuleType("_fake_model_mod")
    mod.PIPELINE_STAGES = STAGES

    class _ModPipe(_Pipe):
        PIPELINE_STAGES = None  # declared on the module instead, as some models do

    _ModPipe.__module__ = mod.__name__
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    assert sm.looks_like_a_pipeline(_ModPipe())
    assert sm.stage_methods(_ModPipe()) == {"alpha": "run_a", "beta": "run_b", "gamma": "run_c"}

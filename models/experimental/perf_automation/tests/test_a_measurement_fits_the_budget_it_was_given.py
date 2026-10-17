# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""How many replays to average is a consequence of what one costs, not a number per model.

16 replay iterations is a noise-reduction choice, and a good one for a step costing milliseconds: the
timer's own jitter is a large fraction of one iteration, so averaging many is how the figure becomes
stable. For a stage costing 68 SECONDS an iteration it buys nothing measurable and costs 18 minutes of
device time per stage.

That is what killed a Qwen-Image-Edit capture. `run_fresh_trace_capture` gives the capture a 900 s
budget; `probes._execute` ends it absolutely at 4x that. The capture needed 16 x 68 s per stage across
several stages, so it ran past the wall and was SIGKILLed mid-progress -- one stage traced, the next
mid-replay -- twice, 3611 s apart, by the gate's own process. The retry bought the same wall again.

The caller already states the constraint and the child inherits it (`_run_perf_node` copies
os.environ), so the measurement is now sized to fit the budget: one replay is timed -- its own time is
a valid steady-state sample, the trace being already captured and warmed -- and only as many more as
the budget affords are added. At least two, so there is always an average; never more than asked for,
so a cheap stage is measured exactly as before; unchanged entirely when no budget is stated.
"""

from __future__ import annotations

import os
import sys
import types


def _tr(monkeypatch):
    monkeypatch.setitem(sys.modules, "ttnn", types.SimpleNamespace())
    sys.modules.pop("agent.trace_replay", None)
    import importlib

    _here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    return importlib.import_module("agent.trace_replay")


# --- the count follows the cost -------------------------------------------------------------------


def test_an_expensive_stage_is_measured_in_what_it_was_given(monkeypatch):
    """THE FAILURE: 16 x 68s per stage put the capture past its budget and into the kill."""
    TR = _tr(monkeypatch)
    n = TR._affordable_iters(16, 68.0, 225.0)
    assert 2 <= n < 16
    assert n * 68.0 <= 225.0, "the measurement must fit the share it was handed"


def test_a_cheap_stage_is_measured_exactly_as_before(monkeypatch):
    """Averaging many is how a millisecond step gets a stable number; nothing is taken from it."""
    TR = _tr(monkeypatch)
    assert TR._affordable_iters(16, 0.003, 225.0) == 16


def test_an_average_always_has_at_least_two_samples(monkeypatch):
    """Even a stage so costly that the budget affords one: one sample is not an average."""
    TR = _tr(monkeypatch)
    assert TR._affordable_iters(16, 10_000.0, 225.0) == TR._MIN_REPLAY_ITERS


def test_it_never_runs_more_than_asked_for(monkeypatch):
    TR = _tr(monkeypatch)
    assert TR._affordable_iters(4, 0.001, 10_000.0) == 4
    assert TR._affordable_iters(1, 0.001, 10_000.0) == 1, "a caller asking for one gets one"


def test_with_no_budget_stated_nothing_changes(monkeypatch):
    """The old behaviour is the default: this only ever narrows when a budget was actually given."""
    TR = _tr(monkeypatch)
    assert TR._affordable_iters(16, 68.0, 0) == 16
    assert TR._affordable_iters(16, 68.0, None) == 16


def test_an_unmeasured_cost_never_narrows_anything(monkeypatch):
    TR = _tr(monkeypatch)
    assert TR._affordable_iters(16, 0, 900.0) == 16
    assert TR._affordable_iters(16, -1.0, 900.0) == 16


def test_a_nonsense_request_does_not_raise(monkeypatch):
    TR = _tr(monkeypatch)
    assert TR._affordable_iters(None, 1.0, 100.0) >= 1
    assert TR._affordable_iters("x", 1.0, 100.0) >= 1


# --- the budget comes from the caller, split across the stages ------------------------------------


def test_the_budget_is_read_from_what_the_caller_stated(monkeypatch):
    TR = _tr(monkeypatch)
    for k in TR._REPLAY_BUDGET_ENVS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PERF_MCP_VALIDATE_TIMEOUT", "900")
    assert TR._measurement_budget_s(1) == 900.0


def test_each_stage_gets_a_share_not_the_whole_budget(monkeypatch):
    """They run in sequence: a stage that spends everything starves the ones after it."""
    TR = _tr(monkeypatch)
    for k in TR._REPLAY_BUDGET_ENVS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PERF_MCP_VALIDATE_TIMEOUT", "900")
    assert TR._measurement_budget_s(4) == 225.0
    assert TR._measurement_budget_s(0) == 900.0  # a degenerate count must not divide by zero


def test_nothing_stated_means_no_budget(monkeypatch):
    TR = _tr(monkeypatch)
    for k in TR._REPLAY_BUDGET_ENVS:
        monkeypatch.delenv(k, raising=False)
    assert TR._measurement_budget_s(4) == 0.0


def test_a_malformed_budget_is_ignored_not_raised(monkeypatch):
    TR = _tr(monkeypatch)
    for k in TR._REPLAY_BUDGET_ENVS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PERF_MCP_VALIDATE_TIMEOUT", "not a number")
    assert TR._measurement_budget_s(1) == 0.0


# --- the reported number must describe what actually ran -----------------------------------------


def test_the_average_divides_by_what_ran_not_by_what_was_asked(monkeypatch):
    """Dividing by the request would report a per-iteration time smaller than any iteration took."""
    import ast
    import inspect
    import textwrap

    TR = _tr(monkeypatch)
    for fn in (TR._measure_native, TR._replay_1cq):
        body = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0])
        assert "/ _REPLAY_ITERS" not in body, f"{fn.__name__} still divides by the requested count"


def test_the_stage_loop_hands_each_stage_its_share(monkeypatch):
    import ast
    import inspect
    import textwrap

    TR = _tr(monkeypatch)
    body = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(TR.measure_adapter))).body[0])
    assert "_measurement_budget_s(len(stages))" in body
    assert "_measure_stage(device, st, _stage_budget)" in body


def test_existing_callers_of_the_stage_measurement_still_work(monkeypatch):
    """The budget is an ADDITIVE parameter; a caller that does not pass one behaves as before."""
    import inspect

    TR = _tr(monkeypatch)
    for fn in (TR._measure_stage, TR._measure_native, TR._replay_1cq):
        assert inspect.signature(fn).parameters["budget_s"].default == 0.0


# --- constraints ----------------------------------------------------------------------------------


def test_it_names_no_model_or_stage(monkeypatch):
    import ast
    import inspect
    import textwrap

    TR = _tr(monkeypatch)
    for fn in (TR._affordable_iters, TR._measurement_budget_s):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        node = tree.body[0]
        if ast.get_docstring(node) is not None:
            node.body = node.body[1:]
        lowered = ast.unparse(node).lower()
        for name in ("qwen", "denoise", "prefill", "vision", "encoder", "decoder", "vae"):
            assert name not in lowered, f"{name!r} in {fn.__name__} assumes the model's vocabulary"


# --- a replay a perf test installed before the budget existed -------------------------------------


def test_a_three_argument_replay_a_test_installed_is_still_called(monkeypatch):
    """Qwen-Image-Edit 2026-10-01: its test replaces _replay_1cq with a 3-argument progress replay;
    called with the budget it raised TypeError in every stage and the whole timing read 0."""
    TR = _tr(monkeypatch)
    seen = []
    monkeypatch.setattr(TR, "_replay_1cq", lambda dev, tid, iters: seen.append(iters) or 0.5)
    assert TR._replay("dev", 7, 4, 120.0) == 0.5
    assert seen == [4]


def test_a_replay_that_takes_the_budget_gets_it(monkeypatch):
    TR = _tr(monkeypatch)
    seen = []
    monkeypatch.setattr(TR, "_replay_1cq", lambda dev, tid, iters, budget_s=0.0: seen.append(budget_s) or 0.25)
    assert TR._replay("dev", 7, 4, 120.0) == 0.25
    assert seen == [120.0]


def test_the_stage_measurement_goes_through_the_tolerant_call(monkeypatch):
    import ast
    import inspect
    import textwrap

    TR = _tr(monkeypatch)
    body = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(TR._measure_stage))).body[0])
    assert "_replay(device, tid, _REPLAY_ITERS, budget_s)" in body
    assert "_replay_1cq(" not in body, "a direct call is the one that broke installed replays"

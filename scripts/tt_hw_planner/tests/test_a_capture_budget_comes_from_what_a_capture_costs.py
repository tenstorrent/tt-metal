# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What a capture may cost is sized from what one has cost, not from a number in the signature.

`run_fresh_trace_capture` carried a literal `timeout_s=900`, and `probes._execute` ends a step
absolutely at `_HARD_CEILING_MULT` x its budget -- so 900 was really a 3600 s wall. A Qwen-Image-Edit
capture needed longer and was SIGKILLed there twice, 3611 s apart, by the gate's own process, both
times with one stage traced and the next mid-replay. The ceiling is RIGHT to be absolute: it is the
only guard against work that progresses forever, and making it relative to quiet time turns it into
dead code (tried, reverted). What was wrong is that it multiplied a number typed for a much smaller
step.

So the sizing policy gets one owner -- `probes.sized_budget`: operator's value, else headroom over
measured cost, else the caller's floor -- and the capture's observation is persisted beside the
demo's other gate state, because the gate runs in a fresh interpreter every round and an in-memory
record would never be read back.
"""

from __future__ import annotations

import json

import pytest

from models.experimental.perf_automation.agent import probes as PR
from scripts.tt_hw_planner import trace_gate as TG

_RUN_ENV = "PERF_MCP_RUN_ID"


# --- the policy, with one owner -------------------------------------------------------------------


def test_a_measurement_beats_the_floor():
    assert PR.sized_budget(1000.0, 900) == 4000
    assert PR.sized_budget(0, 900) == 900, "no measurement -> the floor stands"


def test_the_floor_is_never_undercut():
    """A WRAPPER MUST NEVER BE TIGHTER THAN WHAT IT WRAPS: a small measurement cannot shrink it."""
    assert PR.sized_budget(10.0, 900) == 900


def test_garbage_inputs_do_not_raise():
    assert PR.sized_budget(None, None) == 0
    assert PR.sized_budget("x", "y") == 0


def test_a_caller_may_state_its_own_multiple_and_ceiling():
    """The pre-existing backstop uses 3x and a ceiling; unifying must not have changed it."""
    assert PR.sized_budget(1000.0, 3600, mult=3, ceiling_s=10800) == 3600
    assert PR.sized_budget(4000.0, 3600, mult=3, ceiling_s=10800) == 10800
    assert PR.sized_budget(1000.0, 3600, mult=3, ceiling_s=100) == 3600, "a ceiling below the floor yields the floor"


def test_the_pre_existing_backstop_stopped_re_spelling_the_arithmetic():
    """Constraint: adaptive_backstop had this same max(floor, mult*observed) inline, in this file."""
    import ast
    import inspect
    import textwrap

    body = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(PR.adaptive_backstop))).body[0])
    assert "sized_budget(" in body
    assert "mult * base" not in body, "the arithmetic must not be repeated here"


def test_each_caller_keeps_its_own_override_because_they_differ():
    """One clamps a pinned value to >=1, the other passes it through; flattening that changes meaning."""
    import inspect

    from scripts.tt_hw_planner import cc_harness as CH

    assert "override_env" not in inspect.signature(PR.sized_budget).parameters
    for fn in (CH._gate_status_budget, TG._capture_budget_s):
        body = inspect.getsource(fn)
        assert "sized_budget" in body and "os.environ.get" in body


# --- the capture's own budget ---------------------------------------------------------------------


@pytest.fixture
def demo(tmp_path, monkeypatch):
    monkeypatch.setenv(_RUN_ENV, "run-1")
    monkeypatch.delenv(TG._CAPTURE_BUDGET_ENV, raising=False)
    d = tmp_path / "models" / "demos" / "m"
    d.mkdir(parents=True)
    return d


def test_the_first_capture_gets_the_old_default_as_its_floor(demo):
    assert TG._capture_budget_s(demo) == TG._CAPTURE_FLOOR_S == 900


def test_a_measured_capture_sizes_the_next_one(demo):
    """THE FAILURE: a capture that needed ~3600 s was given a budget whose wall was 3600 s."""
    TG._record_capture_s(demo, 3400.0)
    assert TG._capture_budget_s(demo) > TG._CAPTURE_FLOOR_S
    assert TG._capture_budget_s(demo) >= 3400 * 4


def test_only_the_longest_is_kept(demo):
    """A budget that shrank on a lucky attempt would kill the next one."""
    TG._record_capture_s(demo, 3400.0)
    TG._record_capture_s(demo, 12.0)
    assert TG._observed_capture_s(demo) == 3400.0


def test_a_cost_from_another_run_is_not_reused(demo, monkeypatch):
    """A cost measured on another board says nothing here -- the same rule as the pass cache."""
    TG._record_capture_s(demo, 3400.0)
    monkeypatch.setenv(_RUN_ENV, "run-2")
    assert TG._observed_capture_s(demo) == 0.0
    assert TG._capture_budget_s(demo) == TG._CAPTURE_FLOOR_S


def test_an_operator_override_still_wins(demo, monkeypatch):
    TG._record_capture_s(demo, 3400.0)
    monkeypatch.setenv(TG._CAPTURE_BUDGET_ENV, "123")
    assert TG._capture_budget_s(demo) == 123


def test_a_corrupt_or_missing_record_is_ignored_not_raised(demo):
    assert TG._observed_capture_s(demo) == 0.0
    (demo / TG._CAPTURE_COST_FILE).write_text("{not json")
    assert TG._observed_capture_s(demo) == 0.0
    TG._record_capture_s(demo / "nonexistent", 10.0)  # must not raise


def test_a_nonsense_cost_is_not_recorded(demo):
    TG._record_capture_s(demo, 0)
    TG._record_capture_s(demo, -5)
    assert TG._observed_capture_s(demo) == 0.0


# --- the wiring -----------------------------------------------------------------------------------


def test_the_signature_no_longer_carries_a_typed_budget():
    import inspect

    assert inspect.signature(TG.run_fresh_trace_capture).parameters["timeout_s"].default is None


def test_an_explicit_caller_value_is_still_honoured():
    import ast
    import inspect
    import textwrap

    body = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(TG.run_fresh_trace_capture))).body[0])
    assert "int(timeout_s) if timeout_s else _capture_budget_s(demo_dir)" in body


def test_every_attempt_records_what_it_cost():
    """Including a FAILED one: a capture killed at the wall is the best evidence the wall is too low."""
    import ast
    import inspect
    import textwrap

    body = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(TG.run_fresh_trace_capture))).body[0])
    assert "_record_capture_s(demo_dir" in body
    assert "finally" in inspect.getsource(TG.run_fresh_trace_capture)


def test_it_names_no_model_or_stage():
    import ast
    import inspect
    import textwrap

    for fn in (TG._capture_budget_s, TG._observed_capture_s, TG._record_capture_s, PR.sized_budget):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        node = tree.body[0]
        if ast.get_docstring(node) is not None:
            node.body = node.body[1:]
        lowered = ast.unparse(node).lower()
        for name in ("qwen", "denoise", "prefill", "vision", "vae", "demos/"):
            assert name not in lowered, f"{name!r} in {fn.__name__} assumes the model"

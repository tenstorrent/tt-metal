# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A trace hang already resets the board -- so try again on it before losing the round.

perf_test_gen._run_perf_node catches the hang, runs tt-smi -r, and returns the WEDGE. The retry loop
that follows such a reset lives in `generate_perf_test` (bounded by _TRACE_WEDGE_LIMIT); the gate
reaches the capture through `validate_generated_perf_test`, which has none. So the board was reset
and the result discarded.

Reaching the capture at all means the correctness gate has just run -- ~3.5 h of device work on the
Qwen-Image-Edit bring-up. Six rounds over two days each paid that, wedged once, and stopped. One more
~10-minute capture on a just-reset board is nearly free by comparison.

_is_device_disruption argues the other way -- "that already got one reset and must return to the
caller as a WEDGE, NOT loop reset+retry (which just re-hangs)" -- which is why this is ONE extra
attempt by default rather than a loop, and why every attempt's detail is kept: if it re-hangs the
report says so with both attempts' evidence, which is itself the answer.
"""

from __future__ import annotations

import pytest

from scripts.tt_hw_planner import trace_gate as TG

_WEDGE = "WEDGE: tracy run made no forward progress for 300s; killed process group + tt-smi -r (reset_ok=True)"


@pytest.fixture
def demo(tmp_path, monkeypatch):
    d = tmp_path / "models" / "demos" / "m"
    (d / "tests" / "e2e").mkdir(parents=True)
    (d / "tests" / "e2e" / "test_m_perf.py").write_text("def test_m_perf():\n    pass\n")
    monkeypatch.delenv(TG._WEDGE_RETRY_ENV, raising=False)
    monkeypatch.setattr(TG, "read_trace_caps", lambda _d: None)
    return d


def _capture(monkeypatch, results):
    """Stand in for validate_generated_perf_test, yielding one result per call."""
    calls = []

    def _fake(perf, task, **kw):
        calls.append(perf)
        return results[min(len(calls) - 1, len(results) - 1)]

    import models.experimental.perf_automation.agent.perf_test_gen as PG

    monkeypatch.setattr(PG, "validate_generated_perf_test", _fake)
    return calls


def test_the_default_is_three_attempts(demo, monkeypatch):
    """Sized from cost: ~5-15 min a capture against a ~3.5 h round, and the value decays after the
    third (establish -> did the reset help -> flaky or deterministic)."""
    assert TG._WEDGE_RETRIES == 2
    calls = _capture(monkeypatch, [("invalid", _WEDGE)])
    TG.run_fresh_trace_capture(demo)
    assert len(calls) == 3


def test_every_attempt_says_WHY_it_failed(demo, monkeypatch):
    """The feedback requirement: each attempt carries its own stage evidence, so 'same place twice'
    and 'got further' are distinguishable. The stage lines survive _extract_error's whitelist."""
    a1 = _WEDGE + " TRACE_STAGE_MS[alpha]=1.0 path=trace+1cq TRACE_STAGE_BYTES[beta]=9"
    a2 = _WEDGE + " TRACE_STAGE_MS[alpha]=1.0 path=trace+1cq TRACE_STAGE_MS[beta]=2.0 path=trace+1cq"
    _capture(monkeypatch, [("invalid", a1), ("invalid", a2), ("invalid", a2)])
    _caps, detail = TG.run_fresh_trace_capture(demo)
    assert "attempt 1:" in detail and "attempt 2:" in detail and "attempt 3:" in detail
    assert "TRACE_STAGE_BYTES[beta]" in detail, "attempt 1's stalled stage must be visible"
    assert "TRACE_STAGE_MS[beta]" in detail, "attempt 2 getting further must be visible"


def test_a_wedge_is_retried_on_the_reset_board(demo, monkeypatch):
    """THE FIX: one wedge used to end the round. Now the reset board gets another capture."""
    calls = _capture(monkeypatch, [("invalid", _WEDGE), ("ok_1cq", "")])
    _caps, detail = TG.run_fresh_trace_capture(demo)
    assert len(calls) == 2, "the reset board must get another attempt"
    assert "attempt 1:" in detail and "attempt 2:" in detail
    assert "ok_1cq" in detail


def test_if_it_just_re_hangs_both_attempts_are_reported(demo, monkeypatch):
    """The case the original comment predicts. The cost is one capture; the payoff is the evidence."""
    calls = _capture(
        monkeypatch,
        [("invalid", _WEDGE + " froze in alpha"), ("invalid", _WEDGE + " froze in beta")],
    )
    _caps, detail = TG.run_fresh_trace_capture(demo)
    assert len(calls) == 3, "it exhausts the bound rather than stopping at the first re-hang"
    assert "froze in alpha" in detail and "froze in beta" in detail, "the progression must survive"


def test_an_ordinary_failure_is_NOT_retried(demo, monkeypatch):
    """A code error is the agent's to fix -- retrying it would just burn the device twice."""
    calls = _capture(monkeypatch, [("invalid", "E   AssertionError: pcc 0.8 < 0.99")])
    _caps, detail = TG.run_fresh_trace_capture(demo)
    assert len(calls) == 1
    assert "attempt" not in detail, "a single attempt reports as before, unprefixed"


def test_a_success_is_not_retried(demo, monkeypatch):
    calls = _capture(monkeypatch, [("ok_1cq", "")])
    TG.run_fresh_trace_capture(demo)
    assert len(calls) == 1


def test_the_bound_is_configurable_and_zero_restores_the_old_behaviour(demo, monkeypatch):
    monkeypatch.setenv(TG._WEDGE_RETRY_ENV, "0")
    calls = _capture(monkeypatch, [("invalid", _WEDGE)])
    TG.run_fresh_trace_capture(demo)
    assert len(calls) == 1, "0 retries must behave exactly as before this change"
    monkeypatch.setenv(TG._WEDGE_RETRY_ENV, "4")
    calls = _capture(monkeypatch, [("invalid", _WEDGE)])
    TG.run_fresh_trace_capture(demo)
    assert len(calls) == 5


def test_a_junk_bound_falls_back_instead_of_raising(monkeypatch):
    monkeypatch.setenv(TG._WEDGE_RETRY_ENV, "not a number")
    assert TG._wedge_retries() == TG._WEDGE_RETRIES


def test_a_raising_capture_stops_immediately(demo, monkeypatch):
    import models.experimental.perf_automation.agent.perf_test_gen as PG

    calls = []

    def _boom(perf, task, **kw):
        calls.append(perf)
        raise RuntimeError("device gone")

    monkeypatch.setattr(PG, "validate_generated_perf_test", _boom)
    _caps, detail = TG.run_fresh_trace_capture(demo)
    assert len(calls) == 1 and "capture raised" in detail


def test_no_perf_test_is_unchanged(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert TG.run_fresh_trace_capture(d) == (None, "no perf test to capture")


def test_the_retry_is_announced(demo, monkeypatch, capsys):
    _capture(monkeypatch, [("invalid", _WEDGE), ("ok_1cq", "")])
    TG.run_fresh_trace_capture(demo)
    out = capsys.readouterr().out
    assert "capture wedged and the board was reset" in out


def test_it_names_no_model_or_stage():
    import ast
    import inspect
    import textwrap

    for fn in (TG.run_fresh_trace_capture, TG._is_wedge, TG._wedge_retries):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        node = tree.body[0]
        if ast.get_docstring(node) is not None:
            node.body = node.body[1:]
        lowered = ast.unparse(node).lower()
        for name in ("qwen", "denoise", "vae", "prefill", "encoder", "vision"):
            assert name not in lowered, f"{name!r} in {fn.__name__}"

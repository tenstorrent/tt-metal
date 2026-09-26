# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The out-of-band gate check may not be tighter than the gate it wraps.

`gate_status` ran the gate in a subprocess under a flat 3600 s default while the gate's own budget
was 14400, and swallowed the TimeoutExpired into `{"can_stop": False, "reason": ""}`. On a T3K at
batch 32 the e2e gate needs ~110 min, so four consecutive rounds were SIGKILLed at 59m41s having
done real work -- each reported as an ordinary "not done" with no reason, indistinguishable from a
PCC failure. It had never shown up before because the same gate at batch 4 takes ~15 min; enforcing
`--batch` is what made the work exceed the wrapper.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from scripts.tt_hw_planner import cc_harness as H


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.delenv(H._GATE_STATUS_TIMEOUT_ENV, raising=False)
    H._gate_status_observed.clear()
    yield
    H._gate_status_observed.clear()


# --- the budget ----------------------------------------------------------------------------------


def test_the_wrapper_is_never_tighter_than_what_it_wraps():
    """The exact T3K shape: a gate budgeted 4h, checked under a 1h wrapper."""
    assert H._gate_status_budget("e2e_mcp", 14400) >= 14400


def test_an_operator_value_wins_and_is_not_scaled():
    os.environ[H._GATE_STATUS_TIMEOUT_ENV] = "900"
    try:
        assert H._gate_status_budget("e2e_mcp", 14400) == 900
    finally:
        del os.environ[H._GATE_STATUS_TIMEOUT_ENV]


def test_a_junk_override_falls_back_rather_than_raising():
    os.environ[H._GATE_STATUS_TIMEOUT_ENV] = "not-a-number"
    try:
        assert H._gate_status_budget("e2e_mcp", 14400) >= 14400
    finally:
        del os.environ[H._GATE_STATUS_TIMEOUT_ENV]


def test_the_budget_adapts_to_what_the_gate_actually_costs():
    """Sized from measured cost, so it follows the box/batch/precision without a constant each."""
    H._record_gate_status_cost("e2e_mcp", 6000.0)  # a ~100 min gate, as measured on T3K at B=32
    assert H._gate_status_budget("e2e_mcp", 1) == 4 * 6000


def test_the_budget_only_ever_grows():
    """A gate that ran long once is not evidence the next is hung."""
    H._record_gate_status_cost("e2e_mcp", 6000.0)
    H._record_gate_status_cost("e2e_mcp", 60.0)  # a later, faster gate
    assert H._gate_status_budget("e2e_mcp", 1) == 4 * 6000


def test_each_server_is_costed_separately():
    H._record_gate_status_cost("e2e_mcp", 6000.0)
    assert H._gate_status_budget("bringup_mcp", 1) == 1  # floor only; no history of its own


def test_a_zero_or_negative_duration_is_not_recorded():
    H._record_gate_status_cost("e2e_mcp", 0.0)
    H._record_gate_status_cost("e2e_mcp", -5.0)
    assert H._gate_status_budget("e2e_mcp", 1) == 1


# --- what a timeout reports ------------------------------------------------------------------------


def _gate(monkeypatch, *, raises=None, stdout="CANSTOP=True\n", record=None):
    def _run(cmd, **kw):
        if record is not None:
            record["timeout"] = kw.get("timeout")
        if raises:
            raise raises
        return subprocess.CompletedProcess(cmd, 0, stdout, "")

    monkeypatch.setattr(H.subprocess, "run", _run)
    return H.gate_status("py", "/dir", "e2e_mcp", {}, "/cwd", timeout_s=14400)


def test_a_killed_gate_says_so_instead_of_returning_a_blank(monkeypatch):
    """The bug: a kill and a failure were the same answer, so the loop retried the same work."""
    got = _gate(monkeypatch, raises=subprocess.TimeoutExpired(cmd="py", timeout=14400))
    assert got["can_stop"] is False
    assert got["reason"], "a killed gate must not come back with an empty reason"
    assert "killed" in got["reason"] and "14400" in got["reason"]
    assert H._GATE_STATUS_TIMEOUT_ENV in got["reason"]  # it says how to raise the limit


def test_the_caller_budget_actually_reaches_subprocess(monkeypatch):
    """Regression: emit-e2e passed nothing, so the 3600 default silently applied."""
    rec = {}
    _gate(monkeypatch, record=rec)
    assert rec["timeout"] >= 14400


def test_a_completed_gate_is_costed_so_the_next_check_is_sized_from_it(monkeypatch):
    _gate(monkeypatch)
    assert "e2e_mcp" in H._gate_status_observed


def test_other_failures_are_unchanged(monkeypatch):
    """A crash is still a quiet not-done -- only the TIMEOUT path gained a reason."""
    got = _gate(monkeypatch, raises=OSError("boom"))
    assert got == {"can_stop": False, "halt": False, "reason": "", "next_op": "", "next_rung": ""}


def test_a_normal_verdict_still_parses(monkeypatch):
    got = _gate(monkeypatch, stdout="CANSTOP=True\nHALT=False\nNEXTOP=x\nGRAD=a,b\n")
    assert got["can_stop"] is True and got["next_op"] == "x" and got["graduated"] == ["a", "b"]


def test_existing_callers_without_a_timeout_are_unchanged(monkeypatch):
    """bringup_cc calls gate_status with no timeout; its 3600 default still applies."""
    rec = {}
    monkeypatch.setattr(
        H.subprocess,
        "run",
        lambda cmd, **kw: (rec.update(timeout=kw.get("timeout")), subprocess.CompletedProcess(cmd, 0, "", ""))[1],
    )
    H.gate_status("py", "/dir", "bringup_mcp", {}, "/cwd")
    assert rec["timeout"] == 3600


def test_emit_e2e_hands_the_gate_its_own_budget():
    """The structural fix, asserted at the call site itself."""
    import inspect

    from scripts.tt_hw_planner.commands import emit_e2e as E

    src = inspect.getsource(E._run_emit_e2e_cc)
    assert "gate_status(" in src
    assert "timeout_s=timeout_s" in src, "the wrapper must be handed the gate's own budget"

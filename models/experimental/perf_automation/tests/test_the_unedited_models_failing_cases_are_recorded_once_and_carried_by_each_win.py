"""The correctness file's failing cases on the UNEDITED model are recorded before the first round,
handed to every check_pcc, reused while HEAD stands, and replaced by a banked win's own list.

Not an MCP tool: the agent cannot move the bar. A run that produced no PCC records nothing, so
the strict rule (tolerate nothing) stays in force."""

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))


@pytest.fixture()
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_KERNEL_LOG", str(tmp_path / "kl.json"))
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    monkeypatch.setattr(m, "_head_sha_quiet", lambda: "a" * 40)
    return m


def _gate_run(mcp, monkeypatch, **verdict):
    calls = {"n": 0}

    def _fake_run_pcc(ctx):
        calls["n"] += 1
        return dict(verdict)

    monkeypatch.setattr(mcp, "run_pcc", _fake_run_pcc)
    return calls


def test_the_baseline_records_the_failing_cases_and_the_context_reads_them_back(mcp, monkeypatch):
    _gate_run(mcp, monkeypatch, status="ok", pcc=0.999, threshold=0.99, failed_tests=["test_gate2"])
    out = mcp.record_pcc_baseline()
    assert out["recorded"] is True and out["reused"] is False
    assert json.loads(mcp._gate_baseline_tests_path().read_text())["failed_tests"] == ["test_gate2"]
    assert mcp._Ctx().baseline_failed_tests() == ["test_gate2"]


def test_nothing_recorded_means_nothing_tolerated(mcp):
    assert mcp.read_baseline_failed_tests() is None
    assert mcp._Ctx().baseline_failed_tests() is None


def test_a_baseline_run_without_a_pcc_records_nothing(mcp, monkeypatch):
    _gate_run(mcp, monkeypatch, status="crash", error="device fatal", failed_tests=["test_x"])
    out = mcp.record_pcc_baseline()
    assert out["recorded"] is False
    assert mcp.read_baseline_failed_tests() is None


def test_the_record_is_reused_while_head_is_the_sha_it_was_taken_at(mcp, monkeypatch):
    calls = _gate_run(mcp, monkeypatch, status="ok", pcc=0.999, threshold=0.99, failed_tests=[])
    assert mcp.record_pcc_baseline()["reused"] is False
    again = mcp.record_pcc_baseline()
    assert again["reused"] is True and again["failed_tests"] == [] and calls["n"] == 1
    monkeypatch.setattr(mcp, "_head_sha_quiet", lambda: "b" * 40)
    assert mcp.record_pcc_baseline()["reused"] is False and calls["n"] == 2


def test_check_pcc_records_the_names_so_the_attempt_can_say_which_case_broke(mcp, monkeypatch):
    _gate_run(
        mcp,
        monkeypatch,
        status="tests_failed",
        pcc=0.999,
        threshold=0.99,
        failed_tests=["test_gate2", "test_quality"],
        new_failed_tests=["test_quality"],
    )
    monkeypatch.setattr(mcp, "_note_device_ok", lambda *a, **k: None)
    res = mcp.check_pcc()
    assert res["status"] == "tests_failed"
    v = mcp.gate_verdicts()["pcc"]
    assert v["status"] == "tests_failed" and v["new_failed_tests"] == ["test_quality"]
    assert mcp.gates_allow_banking()[0] is False


def test_a_banked_win_makes_its_own_failing_cases_the_new_baseline(mcp, monkeypatch):
    mcp._write_baseline_failed_tests(["test_gate2", "test_flaky"], "a" * 40)
    mcp.record_gate_verdict("pcc", "ok", pcc=0.999, threshold=0.99, failed_tests=["test_gate2"], measurement_id="fp-1")
    mcp._bank_failed_tests_baseline("c" * 40)
    doc = json.loads(mcp._gate_baseline_tests_path().read_text())
    assert doc["failed_tests"] == ["test_gate2"] and doc["sha"] == "c" * 40


def test_a_refused_verdict_does_not_move_the_baseline(mcp, monkeypatch):
    mcp._write_baseline_failed_tests(["test_gate2"], "a" * 40)
    mcp.record_gate_verdict(
        "pcc",
        "tests_failed",
        pcc=0.999,
        threshold=0.99,
        failed_tests=["test_gate2", "test_quality"],
        measurement_id="fp-2",
    )
    mcp._bank_failed_tests_baseline("c" * 40)
    assert mcp.read_baseline_failed_tests() == ["test_gate2"]

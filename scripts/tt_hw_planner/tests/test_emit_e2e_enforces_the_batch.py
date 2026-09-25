# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The e2e gate enforces the `--batch` it was asked for.

`--batch` used to reach only the builder's prompt. A T3K Qwen-Image-Edit run asked for `--batch 32`,
the agent wrote the gate test with `E2E_BATCH = 4`, and the gate -- which ran whatever the test typed --
passed on 4 samples. The 32-sample result (20/32 at the PCC bar) lived only in a README, and the run was
reported as a batch-32 PASS. The gate now hands the tests the batch and reads back what they drove.
"""
from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

from scripts.tt_hw_planner.commands import emit_e2e as E
from models.experimental.perf_automation.agent import perf_adapter as PA


def test_the_report_round_trips_and_the_last_one_wins():
    assert PA.parse_batch_report(PA.batch_report_line(32)) == 32
    assert PA.parse_batch_report("x\n%s\nlog\n%s\n" % (PA.batch_report_line(4), PA.batch_report_line(32))) == 32
    assert PA.parse_batch_report("[e2e] batch=32 steps=50") is None
    assert PA.parse_batch_report("") is None and PA.parse_batch_report(None) is None


def test_the_requested_batch_passes():
    assert E._batch_gate_reason(32, "..." + PA.batch_report_line(32) + "\n1 passed") is None


def test_a_test_that_drove_fewer_samples_fails():
    """The exact T3K shape: asked for 32, the test drove 4."""
    reason = E._batch_gate_reason(32, PA.batch_report_line(4) + "\n1 passed")
    assert reason and "drove 4" in reason and "--batch 32" in reason
    assert PA.BATCH_ENV in reason and PA.batch_report_line(32) in reason  # it says how to fix it


def test_a_test_that_never_reports_its_batch_fails():
    reason = E._batch_gate_reason(32, "[e2e] batch=32 steps=50\n1 passed")
    assert reason and "never reported" in reason


def _demo(tmp_path):
    demo = tmp_path / "models" / "demos" / "m"
    (demo / "tests" / "e2e").mkdir(parents=True)
    (demo / "tests" / "e2e" / "test_e2e_m.py").write_text("def test_e2e():\n    pass\n")
    return demo


def _run_gate(monkeypatch, tmp_path, batch, test_output):
    seen = {}

    def _run(cmd, **k):
        if "pytest" in cmd:
            seen["env"] = dict(k.get("env") or {})
            return subprocess.CompletedProcess(cmd, 0, test_output, "")
        return subprocess.CompletedProcess(cmd, 1, "", "")

    monkeypatch.setenv("E2E_REQUIRE_ON_DEVICE", "0")
    monkeypatch.delenv(PA.BATCH_ENV, raising=False)
    monkeypatch.setattr(E.subprocess, "run", _run)
    ok, reasons = E._run_deterministic_gates(_demo(tmp_path), 0.99, 60, batch=batch)
    return seen["env"], [r for r in reasons if r.startswith("G3 batch")]


def test_the_gate_runs_the_tests_at_the_requested_batch(monkeypatch, tmp_path):
    env, batch_reasons = _run_gate(monkeypatch, tmp_path, 32, PA.batch_report_line(32))
    assert env[PA.BATCH_ENV] == "32"
    assert batch_reasons == []


def test_the_gate_rejects_a_smaller_batch_than_requested(monkeypatch, tmp_path):
    _, batch_reasons = _run_gate(monkeypatch, tmp_path, 32, PA.batch_report_line(4))
    assert len(batch_reasons) == 1 and "drove 4" in batch_reasons[0]


def test_the_default_batch_is_unchanged(monkeypatch, tmp_path):
    """--batch defaults to 1: the tests' own batch is left unchecked and nothing new is set, as before."""
    env, batch_reasons = _run_gate(monkeypatch, tmp_path, 1, "1 passed")
    assert PA.BATCH_ENV not in env
    assert batch_reasons == []


def test_existing_callers_without_a_batch_still_work(monkeypatch, tmp_path):
    monkeypatch.setenv("E2E_REQUIRE_ON_DEVICE", "0")
    monkeypatch.setattr(E.subprocess, "run", lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, "1 passed", ""))
    ok, reasons = E._run_deterministic_gates(_demo(tmp_path), 0.99, 60)
    assert not [r for r in reasons if r.startswith("G3 batch")]


def test_the_builder_is_told_the_same_names_the_gate_checks():
    block = E._batch_prompt_block(32)
    assert "$%s=32" % PA.BATCH_ENV in block
    assert PA.batch_report_line(32) in block
    assert E._BATCH_COMMON_RULES.format(batch=32) in block  # the existing rules are unchanged
    assert E._batch_prompt_block(1) == ""


def test_the_gate_server_passes_the_request_to_the_gate(monkeypatch):
    pytest.importorskip("mcp")
    monkeypatch.setenv(E.E2E_MCP_BATCH_ENV, "32")
    monkeypatch.setenv("E2E_MCP_DEMO_DIR", "/nonexistent-demo")
    sys.modules.pop("scripts.tt_hw_planner.e2e_mcp", None)
    M = importlib.import_module("scripts.tt_hw_planner.e2e_mcp")
    seen = {}
    monkeypatch.setattr(
        M, "_run_deterministic_gates", lambda d, p, t, batch=1: seen.update(batch=batch) or (False, ["x"])
    )
    fn = getattr(M.termination_check, "fn", M.termination_check)
    assert fn()["can_stop"] is False
    assert seen["batch"] == 32
    sys.modules.pop("scripts.tt_hw_planner.e2e_mcp", None)

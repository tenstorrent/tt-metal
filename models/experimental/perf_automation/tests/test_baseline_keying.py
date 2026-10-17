# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The rolling baseline file is KEYED by (model, task) on both sides.

run.py read the unkeyed "perf_mcp_baseline.json" while perf_mcp wrote a keyed one, so the "before"
number came from whatever model last profiled anywhere on the box. llama3_1_8b_p150 reported

    eager per-op device time (all layers):  0.06 ms  ->  648.17 ms   (-1062476.1%, 0.00x)

against a real 648 ms reading, while its own baseline sat in the keyed file at 2464.18 ms. A file
any other process can write is not this run's baseline -- the same defect already fixed for the
full-pipeline scoreboard.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _run_mod():
    spec = importlib.util.spec_from_file_location("run_baseline_keying_ut", _ROOT / "cc_optimize" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_baseline_keying_ut"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_baseline_name_is_keyed_by_model_and_task(monkeypatch):
    run = _run_mod()
    monkeypatch.setenv("PERF_MCP_MODEL_NAME", "llama3_1_8b_p150")
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    assert run._baseline_name() == "perf_mcp_baseline_llama3_1_8b_p150_main.json"
    monkeypatch.setenv("PERF_MCP_TASK", "t2s")
    assert run._baseline_name() == "perf_mcp_baseline_llama3_1_8b_p150_t2s.json"


def test_two_models_do_not_share_a_baseline(monkeypatch):
    run = _run_mod()
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    monkeypatch.setenv("PERF_MCP_MODEL_NAME", "model_a")
    a = run._baseline_name()
    monkeypatch.setenv("PERF_MCP_MODEL_NAME", "model_b")
    assert run._baseline_name() != a


def test_baseline_ms_reads_the_keyed_file_not_the_global(monkeypatch, tmp_path):
    """THE REGRESSION: a stale unkeyed file must not supply this run's anchor."""
    run = _run_mod()
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setattr(run.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv("PERF_MCP_MODEL_NAME", "llama3_1_8b_p150")
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    (tmp_path / "perf_mcp_baseline.json").write_text(json.dumps({"device_ms": 0.06}))
    (tmp_path / "perf_mcp_baseline_llama3_1_8b_p150_main.json").write_text(json.dumps({"device_ms": 2464.18}))
    assert run._baseline_ms() == 2464.18


def test_baseline_ms_is_none_when_this_run_has_no_baseline(monkeypatch, tmp_path):
    """No keyed file must mean NO anchor, never a fallback to someone else's number."""
    run = _run_mod()
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setattr(run.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv("PERF_MCP_MODEL_NAME", "never_profiled")
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    (tmp_path / "perf_mcp_baseline.json").write_text(json.dumps({"device_ms": 0.06}))
    assert run._baseline_ms() is None

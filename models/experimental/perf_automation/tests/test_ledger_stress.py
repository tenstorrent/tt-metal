# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stress the measurement ledger: per-module runs, many reruns, corruption, concurrency, odd names.

The ledger is keyed by (model, task). A per-module optimize sets PERF_MCP_TASK to the MODULE name
(module_optimize.py), so each module must keep its own independent history -- if they collided, one
module's baseline would anchor another's report, which is the exact failure the ledger exists to
prevent, just at module granularity.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("meas_stress_ut", _ROOT / "cc_optimize" / "measurements.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["meas_stress_ut"] = m
    spec.loader.exec_module(m)
    return m


def _tmpdir(m, tmp_path, monkeypatch):
    """Point the KEYED ledger namespace at tmp_path.

    PERF_MCP_LEDGER_DIR is the supported redirect and outranks the process temp dir, so patching
    gettempdir alone no longer moves the namespace. Both are set, so the intent holds whichever the
    code ends up consulting.
    """
    monkeypatch.delenv("PERF_MCP_LEDGER", raising=False)
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setattr(m.tempfile, "gettempdir", lambda: str(tmp_path))
    return m


# --- per-module optimize -----------------------------------------------------------------------


def test_each_module_keeps_its_own_baseline(tmp_path, monkeypatch):
    """PERF_MCP_TASK is the module name in a per-module run. attention's 900 ms baseline must never
    anchor mlp's report."""
    m = _tmpdir(_mod(), tmp_path, monkeypatch)
    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 900.0, depth="1", mode="eager", model="llama", task="attention")
    m.record(m.KIND_EAGER, m.PHASE_AFTER, 400.0, depth="1", mode="eager", model="llama", task="attention")
    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 120.0, depth="1", mode="eager", model="llama", task="mlp")
    m.record(m.KIND_EAGER, m.PHASE_AFTER, 80.0, depth="1", mode="eager", model="llama", task="mlp")

    assert m.first(m.KIND_EAGER, model="llama", task="attention")["value_ms"] == 900.0
    assert m.first(m.KIND_EAGER, model="llama", task="mlp")["value_ms"] == 120.0
    assert m.last(m.KIND_EAGER, model="llama", task="attention")["value_ms"] == 400.0
    assert m.last(m.KIND_EAGER, model="llama", task="mlp")["value_ms"] == 80.0


def test_modules_do_not_share_a_file(tmp_path, monkeypatch):
    m = _tmpdir(_mod(), tmp_path, monkeypatch)
    a = m.ledger_path("llama", "attention")
    b = m.ledger_path("llama", "mlp")
    c = m.ledger_path("llama", "main")
    assert len({a, b, c}) == 3, (a, b, c)


def test_a_module_rerun_keeps_that_modules_original(tmp_path, monkeypatch):
    m = _tmpdir(_mod(), tmp_path, monkeypatch)
    for i, v in enumerate((900.0, 400.0, 380.0, 375.0, 370.0)):
        phase = m.PHASE_BEFORE if i == 0 else m.PHASE_AFTER
        m.record(m.KIND_EAGER, phase, v, depth="1", mode="eager", model="llama", task="attention")
    assert m.first(m.KIND_EAGER, model="llama", task="attention")["value_ms"] == 900.0
    assert m.last(m.KIND_EAGER, model="llama", task="attention")["value_ms"] == 370.0


def test_the_whole_model_run_and_a_module_run_do_not_collide(tmp_path, monkeypatch):
    """A pipeline run (task=main) and a per-module run on the same model coexist."""
    m = _tmpdir(_mod(), tmp_path, monkeypatch)
    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 2464.0, depth="16", mode="eager", model="llama", task="main")
    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 900.0, depth="1", mode="eager", model="llama", task="attention")
    assert m.first(m.KIND_EAGER, model="llama", task="main")["value_ms"] == 2464.0
    assert m.first(m.KIND_EAGER, model="llama", task="attention")["value_ms"] == 900.0


# --- durability under abuse --------------------------------------------------------------------


def test_fifty_reruns_never_move_the_original(tmp_path, monkeypatch):
    m = _tmpdir(_mod(), tmp_path, monkeypatch)
    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 2464.18, depth="16", mode="eager", model="llama")
    for i in range(50):
        m.record(m.KIND_EAGER, m.PHASE_AFTER, 700.0 - i, depth="16", mode="eager", model="llama")
    assert m.first(m.KIND_EAGER, model="llama")["value_ms"] == 2464.18
    assert m.last(m.KIND_EAGER, model="llama")["value_ms"] == 651.0
    assert len(m.rows(m.KIND_EAGER, model="llama")) == 51


def test_a_regression_is_reported_not_hidden(tmp_path, monkeypatch):
    """If a rerun makes things WORSE, the delta must go negative, not clamp or vanish."""
    m = _tmpdir(_mod(), tmp_path, monkeypatch)
    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 100.0, depth="16", mode="eager", model="llama")
    m.record(m.KIND_EAGER, m.PHASE_AFTER, 140.0, depth="16", mode="eager", model="llama")
    d = m.delta_pct(m.first(m.KIND_EAGER, model="llama"), m.last(m.KIND_EAGER, model="llama"))
    assert d is not None and d < 0, d


def test_interleaved_writes_from_two_processes_are_all_kept(tmp_path, monkeypatch):
    """Append mode: two runs writing the same ledger must not lose or truncate each other's rows."""
    m = _tmpdir(_mod(), tmp_path, monkeypatch)
    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 2464.0, depth="16", mode="eager", model="llama")
    for i in range(20):
        m.record(m.KIND_EAGER, m.PHASE_AFTER, 600.0 + i, depth="16", mode="eager", model="llama", source="A")
        m.record(m.KIND_EAGER, m.PHASE_AFTER, 700.0 + i, depth="16", mode="eager", model="llama", source="B")
    rows = m.rows(m.KIND_EAGER, m.PHASE_AFTER, model="llama")
    assert len(rows) == 40
    assert sum(1 for r in rows if r["source"] == "A") == 20
    assert sum(1 for r in rows if r["source"] == "B") == 20


def test_corruption_in_the_middle_does_not_lose_the_original(tmp_path, monkeypatch):
    m = _tmpdir(_mod(), tmp_path, monkeypatch)
    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 2464.18, depth="16", mode="eager", model="llama")
    p = m.ledger_path("llama", "main")
    with p.open("a") as fh:
        fh.write("{truncated\n\n[]\nnot json at all\n")
    m.record(m.KIND_EAGER, m.PHASE_AFTER, 648.17, depth="16", mode="eager", model="llama")
    assert m.first(m.KIND_EAGER, model="llama")["value_ms"] == 2464.18
    assert m.last(m.KIND_EAGER, model="llama")["value_ms"] == 648.17


def test_a_thousand_rows_still_answers_correctly(tmp_path, monkeypatch):
    m = _tmpdir(_mod(), tmp_path, monkeypatch)
    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 2464.18, depth="16", mode="eager", model="llama")
    for i in range(1000):
        m.record(m.KIND_EAGER, m.PHASE_AFTER, 500.0 + (i % 7), depth="16", mode="eager", model="llama")
    assert m.first(m.KIND_EAGER, model="llama")["value_ms"] == 2464.18
    assert len(m.rows(m.KIND_EAGER, model="llama")) == 1001


# --- refusing what must be refused ---------------------------------------------------------------


def test_depth_change_mid_campaign_refuses_the_delta(tmp_path, monkeypatch):
    """A rerun with a different coverage window must not be subtracted from the original."""
    m = _tmpdir(_mod(), tmp_path, monkeypatch)
    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 2464.18, depth="16", mode="eager", model="llama")
    m.record(m.KIND_EAGER, m.PHASE_AFTER, 601.0, depth="all", mode="eager", model="llama")
    ok, why = m.comparable(m.first(m.KIND_EAGER, model="llama"), m.last(m.KIND_EAGER, model="llama"))
    assert not ok and "depth differs" in why


def test_garbage_values_are_never_stored(tmp_path, monkeypatch):
    m = _tmpdir(_mod(), tmp_path, monkeypatch)
    for bad in (0, -1, None, "abc", float("nan")):
        m.record(m.KIND_EAGER, m.PHASE_BEFORE, bad, depth="16", mode="eager", model="llama")
    assert m.first(m.KIND_EAGER, model="llama") is None


def test_a_hostile_model_name_cannot_escape_the_temp_dir(tmp_path, monkeypatch):
    m = _tmpdir(_mod(), tmp_path, monkeypatch)
    p = m.ledger_path("../../etc/passwd", "main")
    assert p.parent == tmp_path, p
    assert "/" not in p.name and ".." not in p.name, p.name


def test_two_models_never_read_each_others_rows(tmp_path, monkeypatch):
    m = _tmpdir(_mod(), tmp_path, monkeypatch)
    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 2464.0, depth="16", mode="eager", model="llama")
    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 22.9, depth="16", mode="eager", model="bge_m3")
    assert m.first(m.KIND_EAGER, model="llama")["value_ms"] == 2464.0
    assert m.first(m.KIND_EAGER, model="bge_m3")["value_ms"] == 22.9

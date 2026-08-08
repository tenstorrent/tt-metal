# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial and randomised stress: forged rows, torn writes, precision, and invariants that must
hold across any sequence of operations.

The ledger is a line-delimited file whose first row anchors every future report. That makes two
things worth attacking: whether a field can FORGE a row (a newline in a string would end the line
early), and whether the anchor can be moved by any ordering of legal operations.
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_MEAS = _ROOT / "cc_optimize" / "measurements.py"


def _mod():
    spec = importlib.util.spec_from_file_location("meas_adv_ut", _MEAS)
    m = importlib.util.module_from_spec(spec)
    sys.modules["meas_adv_ut"] = m
    spec.loader.exec_module(m)
    return m


def _at(m, tmp_path, monkeypatch):
    monkeypatch.delenv("PERF_MCP_LEDGER", raising=False)
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setattr(m.tempfile, "gettempdir", lambda: str(tmp_path))
    return m


def test_a_newline_in_a_field_cannot_forge_a_row(tmp_path, monkeypatch):
    """The file is line-delimited. If a string field were written raw, a source containing a newline
    plus JSON would END the real row early and inject a second one -- letting an attacker (or a
    careless model id) plant a fake 0.01 ms anchor that no later run could dislodge."""
    m = _at(_mod(), tmp_path, monkeypatch)
    forged = (
        '\n{"schema": 1, "kind": "eager_per_op", "phase": "before", "value_ms": 0.01, '
        '"depth": "16", "mode": "eager", "stage": "", "source": "forged"}'
    )
    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 2464.18, depth="16", mode="eager", source=forged, model="llama")
    rows = m.rows(model="llama")
    assert len(rows) == 1, rows
    assert rows[0]["value_ms"] == 2464.18
    assert m.first(m.KIND_EAGER, model="llama")["value_ms"] == 2464.18


def test_control_characters_survive_a_round_trip(tmp_path, monkeypatch):
    m = _at(_mod(), tmp_path, monkeypatch)
    nasty = 'tab\there "quoted" \\ back\\slash \r\n unicode: αβγ 中文 🙂'
    assert m.record(m.KIND_EAGER, m.PHASE_BEFORE, 1.5, depth="16", mode="eager", source=nasty, model="llama")
    r = m.first(m.KIND_EAGER, model="llama")
    assert r["value_ms"] == 1.5 and r["source"] == nasty.strip()


def test_a_process_killed_mid_campaign_leaves_a_readable_ledger(tmp_path):
    """SIGKILL during writes: the last line may be torn, but every completed row must still parse
    and the anchor must survive."""
    script = textwrap.dedent(
        """
        import importlib.util, sys, time
        spec = importlib.util.spec_from_file_location("m", %r)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        m.record(m.KIND_EAGER, m.PHASE_BEFORE, 2464.18, depth="16", mode="eager")
        for i in range(100000):
            m.record(m.KIND_EAGER, m.PHASE_AFTER, 600.0 + (i %% 5), depth="16", mode="eager")
        """
        % str(_MEAS)
    )
    sp = tmp_path / "w.py"
    sp.write_text(script)
    env = dict(os.environ, PERF_MCP_LEDGER=str(tmp_path / "led.jsonl"))
    p = subprocess.Popen([sys.executable, str(sp)], env=env)
    time.sleep(2.0)
    p.send_signal(signal.SIGKILL)
    p.wait(timeout=30)

    m = _mod()
    os.environ["PERF_MCP_LEDGER"] = str(tmp_path / "led.jsonl")
    try:
        rows = m.rows()
        assert rows, "no rows survived the kill"
        assert rows[0]["value_ms"] == 2464.18, "the anchor did not survive"
        assert all(isinstance(r.get("value_ms"), float) for r in rows)
    finally:
        os.environ.pop("PERF_MCP_LEDGER", None)


def test_the_anchor_is_invariant_under_any_legal_sequence(tmp_path, monkeypatch):
    """Randomised: whatever order of kinds, phases, models and tasks arrives, the first before-row
    of a given (model, task, kind) is fixed the moment it is written."""
    m = _at(_mod(), tmp_path, monkeypatch)
    rnd = random.Random(1234)
    models = ["llama", "bge", "seamless"]
    tasks = ["main", "attention", "mlp"]
    kinds = [m.KIND_EAGER, m.KIND_FULLPIPE, m.KIND_TRACE_PASS]
    expected = {}
    for _ in range(600):
        mo, ta, ki = rnd.choice(models), rnd.choice(tasks), rnd.choice(kinds)
        val = round(rnd.uniform(0.5, 5000.0), 4)
        key = (mo, ta, ki)
        phase = m.PHASE_BEFORE if key not in expected else m.PHASE_AFTER
        if m.record(ki, phase, val, depth="16", mode="eager", model=mo, task=ta) and key not in expected:
            expected[key] = val
    for (mo, ta, ki), val in expected.items():
        got = m.first(ki, model=mo, task=ta)
        assert got and got["value_ms"] == val, (mo, ta, ki, val, got)


def test_reads_are_consistent_while_another_process_writes(tmp_path):
    """A report can render while a profile is being recorded. Every read must see only whole rows."""
    script = textwrap.dedent(
        """
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("m", %r)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        m.record(m.KIND_EAGER, m.PHASE_BEFORE, 2464.18, depth="16", mode="eager")
        for i in range(4000):
            m.record(m.KIND_EAGER, m.PHASE_AFTER, 600.0 + (i %% 9), depth="16", mode="eager")
        """
        % str(_MEAS)
    )
    sp = tmp_path / "w.py"
    sp.write_text(script)
    env = dict(os.environ, PERF_MCP_LEDGER=str(tmp_path / "led.jsonl"))
    p = subprocess.Popen([sys.executable, str(sp)], env=env)
    m = _mod()
    os.environ["PERF_MCP_LEDGER"] = str(tmp_path / "led.jsonl")
    try:
        seen_counts = []
        while p.poll() is None:
            rows = m.rows()  # must never raise on a partially written file
            seen_counts.append(len(rows))
            if rows:
                assert rows[0]["value_ms"] == 2464.18
        p.wait(timeout=60)
        assert max(seen_counts or [0]) > 0
        assert m.first(m.KIND_EAGER)["value_ms"] == 2464.18
        assert len(m.rows()) == 4001
    finally:
        os.environ.pop("PERF_MCP_LEDGER", None)


def test_extreme_but_legal_values_round_trip(tmp_path, monkeypatch):
    m = _at(_mod(), tmp_path, monkeypatch)
    for v in (0.0001, 1e-3, 1e6, 86_400_000.0):
        assert m.record(m.KIND_EAGER, m.PHASE_AFTER, v, depth="16", mode="eager", model="x"), v
    vals = [r["value_ms"] for r in m.rows(m.KIND_EAGER, model="x")]
    assert vals == [0.0001, 0.001, 1000000.0, 86400000.0], vals


def test_a_future_schema_row_is_not_mistaken_for_a_current_one(tmp_path, monkeypatch):
    """Forward compatibility: a row written by a newer version must not silently anchor an older
    reader with fields it cannot interpret."""
    m = _at(_mod(), tmp_path, monkeypatch)
    p = m.ledger_path("llama", "main")
    p.write_text(json.dumps({"schema": 99, "kind": "eager_per_op", "phase": "before", "value_ms": 1.0}) + "\n")
    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 2464.18, depth="16", mode="eager", model="llama")
    rows = m.rows(m.KIND_EAGER, m.PHASE_BEFORE, model="llama")
    assert any(r.get("schema") == _mod()._SCHEMA and r["value_ms"] == 2464.18 for r in rows)


def test_empty_and_whitespace_only_files_are_harmless(tmp_path, monkeypatch):
    m = _at(_mod(), tmp_path, monkeypatch)
    p = m.ledger_path("llama", "main")
    p.write_text("\n\n   \n\t\n")
    assert m.rows(model="llama") == []
    assert m.first(m.KIND_EAGER, model="llama") is None
    assert m.record(m.KIND_EAGER, m.PHASE_BEFORE, 10.0, depth="1", mode="eager", model="llama")
    assert m.first(m.KIND_EAGER, model="llama")["value_ms"] == 10.0

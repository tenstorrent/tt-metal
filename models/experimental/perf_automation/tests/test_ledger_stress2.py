# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Second stress pass: real OS concurrency, failure paths, and the states a long campaign reaches.

The first pass covered logic. This one covers the environment the ledger actually lives in -- two
processes writing at once, a /tmp cleaner deleting the file mid-run, a read-only directory, a
profile that FAILED, and the kinds/phases a rerun produces.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_MEAS = _ROOT / "cc_optimize" / "measurements.py"


def _mod():
    spec = importlib.util.spec_from_file_location("meas_stress2_ut", _MEAS)
    m = importlib.util.module_from_spec(spec)
    sys.modules["meas_stress2_ut"] = m
    spec.loader.exec_module(m)
    return m


def _at(m, tmp_path, monkeypatch):
    monkeypatch.delenv("PERF_MCP_LEDGER", raising=False)
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setattr(m.tempfile, "gettempdir", lambda: str(tmp_path))
    return m


def test_two_real_processes_writing_at_once_lose_no_rows(tmp_path):
    """REAL concurrency, not simulated: an optimize run and a module run can profile at the same
    time. Interleaved appends must not tear a line or drop a row."""
    script = textwrap.dedent(
        """
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("m", %r)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        tag = sys.argv[1]
        for i in range(150):
            m.record(m.KIND_EAGER, m.PHASE_AFTER, 100.0 + i, depth="16", mode="eager", source=tag)
        """
        % str(_MEAS)
    )
    sp = tmp_path / "w.py"
    sp.write_text(script)
    env = dict(os.environ, PERF_MCP_LEDGER=str(tmp_path / "led.jsonl"))
    procs = [subprocess.Popen([sys.executable, str(sp), t], env=env) for t in ("A", "B")]
    for p in procs:
        assert p.wait(timeout=120) == 0

    lines = [l for l in (tmp_path / "led.jsonl").read_text().splitlines() if l.strip()]
    parsed = []
    for l in lines:
        parsed.append(json.loads(l))  # a torn line raises here
    assert len(parsed) == 300, len(parsed)
    assert sum(1 for r in parsed if r["source"] == "A") == 150
    assert sum(1 for r in parsed if r["source"] == "B") == 150


def test_the_file_vanishing_mid_run_does_not_crash(tmp_path, monkeypatch):
    """A /tmp cleaner can remove it. Losing history is bad; crashing the optimize run is worse."""
    m = _at(_mod(), tmp_path, monkeypatch)
    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 2464.18, depth="16", mode="eager", model="llama")
    m.ledger_path("llama", "main").unlink()
    assert m.first(m.KIND_EAGER, model="llama") is None
    assert m.record(m.KIND_EAGER, m.PHASE_AFTER, 648.17, depth="16", mode="eager", model="llama") is True
    assert m.last(m.KIND_EAGER, model="llama")["value_ms"] == 648.17


def test_an_unwritable_directory_is_survived(tmp_path, monkeypatch):
    """record() must never raise into the caller: a failed write is a lost row, not a failed run."""
    m = _mod()
    # This exercises the KEYED-directory path, so the suite-wide private-ledger override must be
    # lifted -- with PERF_MCP_LEDGER set, ledger_path returns that file and never resolves a
    # directory at all. The directory itself is pointed at the read-only dir via the supported
    # override, which outranks the process temp dir.
    monkeypatch.delenv("PERF_MCP_LEDGER", raising=False)
    ro = tmp_path / "ro"
    ro.mkdir()
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(ro))
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(ro))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(ro))
    monkeypatch.setattr(m.tempfile, "gettempdir", lambda: str(ro))
    ro.chmod(0o500)
    try:
        assert m.record(m.KIND_EAGER, m.PHASE_BEFORE, 100.0, depth="16", mode="eager", model="x") is False
        assert m.first(m.KIND_EAGER, model="x") is None
    finally:
        ro.chmod(0o700)


def test_a_failed_profile_writes_no_row(tmp_path, monkeypatch):
    """profile_model returns early on a crash or a partial capture. Nothing may reach the ledger --
    a torn measurement must not become the anchor."""
    monkeypatch.setenv("PERF_MCP_MANIFEST", str(tmp_path / "m.json"))
    (tmp_path / "m.json").write_text('{"config": {}, "perf_test_resolved": {"path": "t.py"}}')
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "led.jsonl"))
    spec = importlib.util.spec_from_file_location("pm_fail_ut", _ROOT / "cc_optimize" / "perf_mcp.py")
    pm = importlib.util.module_from_spec(spec)
    sys.modules["pm_fail_ut"] = pm
    spec.loader.exec_module(pm)

    def boom(cq=1):
        raise RuntimeError("device wedged")

    pm._profile_with_zero_row_retry = boom
    r = pm.profile_model()
    assert r.get("ok") is False
    pm._profile_with_zero_row_retry = lambda cq=1: {"device_ms": 500.0, "capture_partial": "dropped markers"}
    r = pm.profile_model()
    assert r.get("ok") is False
    assert not (tmp_path / "led.jsonl").exists(), "a failed profile reached the ledger"


def test_a_rerun_never_records_a_second_before(tmp_path, monkeypatch):
    """Phase is decided from history, so run 2's first reading is an AFTER even though it is that
    run's own baseline. Otherwise the optimized state would overwrite the original."""
    m = _at(_mod(), tmp_path, monkeypatch)
    for v in (2464.18, 648.17, 601.0, 590.0):
        seen = m.first(m.KIND_EAGER, model="llama")
        m.record(
            m.KIND_EAGER,
            m.PHASE_AFTER if seen else m.PHASE_BEFORE,
            v,
            depth="16",
            mode="eager",
            model="llama",
        )
    befores = m.rows(m.KIND_EAGER, m.PHASE_BEFORE, model="llama")
    assert len(befores) == 1 and befores[0]["value_ms"] == 2464.18


def test_the_fullpipe_kind_is_tracked_independently_of_eager(tmp_path, monkeypatch):
    """Two kinds share a file; one reaching its after must not affect the other's phase."""
    m = _at(_mod(), tmp_path, monkeypatch)
    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 2464.0, depth="16", mode="eager", model="llama")
    m.record(m.KIND_EAGER, m.PHASE_AFTER, 648.0, depth="16", mode="eager", model="llama")
    assert m.first(m.KIND_FULLPIPE, model="llama") is None
    m.record(m.KIND_FULLPIPE, m.PHASE_BEFORE, 48.38, depth="all", mode="trace+1cq", model="llama")
    assert m.first(m.KIND_FULLPIPE, model="llama")["value_ms"] == 48.38
    assert m.first(m.KIND_EAGER, model="llama")["value_ms"] == 2464.0


def test_the_trace_pass_kind_round_trips(tmp_path, monkeypatch):
    m = _at(_mod(), tmp_path, monkeypatch)
    m.record(m.KIND_TRACE_PASS, m.PHASE_BEFORE, 33.89, depth="16", mode="tracy-trace", model="llama")
    r = m.first(m.KIND_TRACE_PASS, model="llama")
    assert r["value_ms"] == 33.89 and r["mode"] == "tracy-trace"


def test_a_very_long_model_id_still_produces_a_usable_filename(tmp_path, monkeypatch):
    """HF ids can be long; a name over the filesystem limit would make every write fail silently."""
    m = _at(_mod(), tmp_path, monkeypatch)
    long_id = "org/" + ("a" * 300)
    p = m.ledger_path(long_id, "main")
    assert len(p.name.encode()) <= 255, len(p.name.encode())
    assert m.record(m.KIND_EAGER, m.PHASE_BEFORE, 12.0, depth="1", mode="eager", model=long_id) is True
    assert m.first(m.KIND_EAGER, model=long_id)["value_ms"] == 12.0


def test_stage_is_part_of_comparability(tmp_path, monkeypatch):
    """prefill and decode are different work; their numbers must never be subtracted."""
    m = _at(_mod(), tmp_path, monkeypatch)
    m.record(m.KIND_FULLPIPE, m.PHASE_BEFORE, 48.0, depth="all", mode="trace+1cq", stage="prefill", model="l")
    m.record(m.KIND_FULLPIPE, m.PHASE_AFTER, 22.0, depth="all", mode="trace+1cq", stage="decode", model="l")
    ok, why = m.comparable(m.first(m.KIND_FULLPIPE, model="l"), m.last(m.KIND_FULLPIPE, model="l"))
    assert not ok and "stage differs" in why

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Issue 8: two producers, two stores -- so the report's headline could read empty.

The baseline profile is produced in ONE place and consumed in TWO forms:

  * ``before_loop.py:647`` writes ``profiles/baseline_profile.json``. The report's op table, block
    timings and roofline all come from this file.
  * ``perf_mcp.py::_ledger_record`` writes the ``KIND_EAGER`` anchor. summary.py reads THAT for the
    headline "eager per-op device time" line.

Only the second is a ledger writer, and it fires solely from the agent-invoked ``profile_model``
MCP tool. So a run whose ledger starts empty and which never happens to make that MCP call has a
complete baseline_profile.json and NO anchor -- the report then prints
"not measured (no ledger reading)" for a number it demonstrably measured. On llama3_1_8b_p150 that
also produced three different totals for one profile (120.59 / 152.02 / 178.85), because each
consumer reached for whichever store it knew about.

The baseline IS the "before" by definition, so the producer that computes it must be a ledger
writer too. Recording at the point of production also stamps the depth the profile was actually
taken at, which is what keeps a 2-layer number from later anchoring a 16-layer run.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _ledger(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("meas_under_test", str(_PA / "cc_optimize" / "measurements.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "ledger.jsonl"))
    return m


def _bl():
    import agent.before_loop as bl

    return bl


PROFILE = {"device_ms": 178.85, "perf_layers": "16", "buckets": [{"id": "matmul", "count": 10, "device_ms": 100.0}]}


def _require_helper():
    bl = _bl()
    fn = getattr(bl, "_record_baseline_anchor", None)
    if fn is None:
        pytest.fail(
            "before_loop has no _record_baseline_anchor: it writes baseline_profile.json but never "
            "records a KIND_EAGER anchor, so summary.py's headline reads 'not measured (no ledger "
            "reading)' for a number the run actually measured."
        )
    return fn


def test_record_baseline_anchor_exists():
    assert _require_helper() is not None


def test_baseline_writes_the_eager_anchor(monkeypatch, tmp_path):
    led = _ledger(monkeypatch, tmp_path)
    bl = _bl()
    _require_helper()
    bl._record_baseline_anchor(PROFILE, model="m", task="main")
    row = led.first(led.KIND_EAGER, led.PHASE_BEFORE, model="m", task="main")
    assert row, "no KIND_EAGER 'before' anchor was written by the baseline producer"
    assert float(row["value_ms"]) == 178.85


def test_anchor_carries_the_depth_it_was_measured_at(monkeypatch, tmp_path):
    led = _ledger(monkeypatch, tmp_path)
    bl = _bl()
    _require_helper()
    bl._record_baseline_anchor(PROFILE, model="m", task="main")
    row = led.first(led.KIND_EAGER, led.PHASE_BEFORE, model="m", task="main")
    assert str(row.get("depth")) == "16", (
        "the anchor must be stamped with the depth it was profiled at, or a 2-layer number can "
        "later anchor a 16-layer run"
    )


def test_all_layers_profile_is_stamped_all(monkeypatch, tmp_path):
    led = _ledger(monkeypatch, tmp_path)
    bl = _bl()
    _require_helper()
    bl._record_baseline_anchor({"device_ms": 45.0}, model="m", task="main")
    row = led.first(led.KIND_EAGER, led.PHASE_BEFORE, model="m", task="main")
    assert str(row.get("depth")) == "all"


def test_baseline_does_not_overwrite_an_existing_before(monkeypatch, tmp_path):
    """A rerun appends an AFTER; the original BEFORE must survive (the write-once anchor rule)."""
    led = _ledger(monkeypatch, tmp_path)
    bl = _bl()
    _require_helper()
    bl._record_baseline_anchor(PROFILE, model="m", task="main")
    bl._record_baseline_anchor({**PROFILE, "device_ms": 99.0}, model="m", task="main")
    row = led.first(led.KIND_EAGER, led.PHASE_BEFORE, model="m", task="main")
    assert float(row["value_ms"]) == 178.85, "a rerun overwrote the original BEFORE anchor"


def test_zero_or_missing_ms_is_not_recorded(monkeypatch, tmp_path):
    led = _ledger(monkeypatch, tmp_path)
    bl = _bl()
    _require_helper()
    bl._record_baseline_anchor({"device_ms": 0}, model="m", task="main")
    bl._record_baseline_anchor({}, model="m", task="main")
    assert not led.first(led.KIND_EAGER, led.PHASE_BEFORE, model="m", task="main")


def test_recording_never_raises(monkeypatch, tmp_path):
    """The baseline must be written even if the ledger is unavailable; this is best-effort."""
    bl = _bl()
    _require_helper()
    monkeypatch.setenv("PERF_MCP_LEDGER", "/nonexistent_xyz/deep/path/l.jsonl")
    bl._record_baseline_anchor(PROFILE, model="m", task="main")  # must not raise


def test_before_loop_calls_it_next_to_the_json_write():
    """Wiring: the anchor must be recorded where the profile is produced, not somewhere optional."""
    src = (_PA / "agent" / "before_loop.py").read_text()
    assert "_record_baseline_anchor(" in src
    i_json = src.index('"baseline_profile.json"')
    i_anchor = src.index("_record_baseline_anchor(", i_json)
    assert i_anchor - i_json < 2000, (
        "the anchor is recorded far from the baseline_profile.json write; keep them together so a "
        "future edit cannot produce one without the other"
    )

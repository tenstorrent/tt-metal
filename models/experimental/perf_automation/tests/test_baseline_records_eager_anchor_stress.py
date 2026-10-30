# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""STRESS for issue 8: one measurement, one anchor -- the two stores must not disagree.

The defect was structural: two producers (before_loop's baseline_profile.json and perf_mcp's
KIND_EAGER anchor), only one of which was a ledger writer, and it optional. Consumers then reached
for whichever store they knew about, so a single profile surfaced as three different totals.

  s1  write-once: the FIRST profile is the permanent BEFORE across any number of reruns
  s2  both producers agree -- before_loop and perf_mcp must record identical anchors for one profile
  s3  unusable values (0, negative, NaN, inf, None, junk) can never become the anchor
  s4  depth is always stamped, across every profile shape
  s5  keying: two models / two tasks never contaminate each other's ledger
  s6  concurrency: 50 threads recording the same baseline still leave ONE agreed BEFORE
  s7  best-effort: an unwritable ledger never breaks the baseline
"""

import importlib.util
import json
import sys
import threading
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _led(monkeypatch, path):
    spec = importlib.util.spec_from_file_location("meas_stress", str(_PA / "cc_optimize" / "measurements.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setenv("PERF_MCP_LEDGER", str(path))
    return m


def _bl():
    import agent.before_loop as bl

    return bl


def _prof(ms=178.85, depth="16"):
    p = {"device_ms": ms}
    if depth is not None:
        p["perf_layers"] = depth
    return p


# --------------------------------------------------------------------------- s1
def test_s1_first_profile_is_the_permanent_before(monkeypatch, tmp_path):
    led = _led(monkeypatch, tmp_path / "l.jsonl")
    bl = _bl()
    for i, ms in enumerate([178.85, 99.0, 45.0, 12.5, 200.0]):
        bl._record_baseline_anchor(_prof(ms), model="m", task="main")
    row = led.first(led.KIND_EAGER, led.PHASE_BEFORE, model="m", task="main")
    assert float(row["value_ms"]) == 178.85, "a rerun displaced the original BEFORE"
    afters = [r for r in led.rows(led.KIND_EAGER, led.PHASE_AFTER, "m", "main")]
    assert len(afters) == 4, f"reruns must append AFTERs, got {len(afters)}"


def test_s1_before_count_is_exactly_one(monkeypatch, tmp_path):
    led = _led(monkeypatch, tmp_path / "l.jsonl")
    bl = _bl()
    for _ in range(20):
        bl._record_baseline_anchor(_prof(), model="m", task="main")
    befores = led.rows(led.KIND_EAGER, led.PHASE_BEFORE, "m", "main")
    assert len(befores) == 1, f"{len(befores)} BEFORE anchors; the anchor must be write-once"


# --------------------------------------------------------------------------- s2
def test_s2_both_producers_record_the_same_anchor(monkeypatch, tmp_path):
    """before_loop and perf_mcp must be interchangeable for one profile -- that equivalence is the
    whole point of making the baseline producer a ledger writer."""
    prof = _prof(123.45, "8")

    a = tmp_path / "a.jsonl"
    led_a = _led(monkeypatch, a)
    _bl()._record_baseline_anchor(prof, model="m", task="main")
    row_a = led_a.first(led_a.KIND_EAGER, led_a.PHASE_BEFORE, model="m", task="main")

    b = tmp_path / "b.jsonl"
    led_b = _led(monkeypatch, b)
    led_b.record(
        led_b.KIND_EAGER,
        led_b.PHASE_BEFORE,
        prof["device_ms"],
        depth=str(prof["perf_layers"]),
        mode="eager",
        source="profile_model",
        model="m",
        task="main",
    )
    row_b = led_b.first(led_b.KIND_EAGER, led_b.PHASE_BEFORE, model="m", task="main")

    for k in ("kind", "phase", "value_ms", "depth", "mode", "derived"):
        assert row_a[k] == row_b[k], f"producers disagree on {k}: {row_a[k]!r} vs {row_b[k]!r}"


# --------------------------------------------------------------------------- s3
@pytest.mark.parametrize("ms", [0, 0.0, -1, -999.9, None, "", "abc", [], {}, float("nan"), float("inf"), float("-inf")])
def test_s3_unusable_values_never_anchor(monkeypatch, tmp_path, ms):
    led = _led(monkeypatch, tmp_path / "l.jsonl")
    bl = _bl()
    bl._record_baseline_anchor({"device_ms": ms, "perf_layers": "16"}, model="m", task="main")
    assert not led.first(
        led.KIND_EAGER, led.PHASE_BEFORE, model="m", task="main"
    ), f"{ms!r} became the permanent anchor; no later run could dislodge it"


def test_s3_a_good_value_after_bad_ones_still_becomes_the_before(monkeypatch, tmp_path):
    led = _led(monkeypatch, tmp_path / "l.jsonl")
    bl = _bl()
    for bad in (0, -1, float("nan"), None):
        bl._record_baseline_anchor({"device_ms": bad}, model="m", task="main")
    bl._record_baseline_anchor(_prof(50.0), model="m", task="main")
    row = led.first(led.KIND_EAGER, led.PHASE_BEFORE, model="m", task="main")
    assert row and float(row["value_ms"]) == 50.0


# --------------------------------------------------------------------------- s4
@pytest.mark.parametrize("depth", ["16", "2", "all", "1", None, "", 32])
def test_s4_depth_is_always_stamped(monkeypatch, tmp_path, depth):
    led = _led(monkeypatch, tmp_path / "l.jsonl")
    bl = _bl()
    prof = {"device_ms": 10.0}
    if depth is not None:
        prof["perf_layers"] = depth
    bl._record_baseline_anchor(prof, model="m", task="main")
    row = led.first(led.KIND_EAGER, led.PHASE_BEFORE, model="m", task="main")
    assert row, f"depth={depth!r} produced no anchor"
    assert str(row["depth"]).strip(), "an unstamped anchor is exactly what the report cannot use"


# --------------------------------------------------------------------------- s5
def test_s5_models_and_tasks_do_not_contaminate(monkeypatch, tmp_path):
    bl = _bl()
    seen = {}
    for model, task, ms in [("a", "main", 1.0), ("b", "main", 2.0), ("a", "decode", 3.0)]:
        p = tmp_path / f"{model}_{task}.jsonl"
        led = _led(monkeypatch, p)
        bl._record_baseline_anchor(_prof(ms), model=model, task=task)
        seen[(model, task)] = float(led.first(led.KIND_EAGER, led.PHASE_BEFORE, model=model, task=task)["value_ms"])
    assert seen == {("a", "main"): 1.0, ("b", "main"): 2.0, ("a", "decode"): 3.0}


# --------------------------------------------------------------------------- s6
def test_s6_concurrent_baseline_writes_leave_one_agreed_before(monkeypatch, tmp_path):
    led = _led(monkeypatch, tmp_path / "l.jsonl")
    bl = _bl()
    errors = []

    def worker():
        try:
            bl._record_baseline_anchor(_prof(178.85), model="m", task="main")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=worker) for _ in range(50)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors, f"concurrent recording raised: {errors[:3]}"
    befores = led.rows(led.KIND_EAGER, led.PHASE_BEFORE, "m", "main")
    assert befores, "no BEFORE anchor survived concurrent writes"
    assert (
        len({r["value_ms"] for r in befores}) == 1
    ), f"concurrent writers disagreed on the BEFORE value: {[r['value_ms'] for r in befores]}"


def test_s6_ledger_lines_stay_valid_json_under_concurrency(monkeypatch, tmp_path):
    p = tmp_path / "l.jsonl"
    _led(monkeypatch, p)
    bl = _bl()
    ts = [
        threading.Thread(target=lambda: bl._record_baseline_anchor(_prof(1.0), model="m", task="main"))
        for _ in range(50)
    ]
    [t.start() for t in ts]
    [t.join() for t in ts]
    for i, line in enumerate(p.read_text().splitlines()):
        if line.strip():
            json.loads(line)  # interleaved appends must not tear a row


# --------------------------------------------------------------------------- s7
@pytest.mark.parametrize("path", ["/nonexistent_xyz/a/b/l.jsonl", "/dev/null/l.jsonl", "/proc/l.jsonl"])
def test_s7_unwritable_ledger_never_raises(monkeypatch, path):
    bl = _bl()
    monkeypatch.setenv("PERF_MCP_LEDGER", path)
    bl._record_baseline_anchor(_prof(), model="m", task="main")


def test_s7_malformed_profile_never_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    bl = _bl()
    for bad in (None, {}, [], "string", 42, {"device_ms": {"nested": 1}}):
        bl._record_baseline_anchor(bad, model="m", task="main")

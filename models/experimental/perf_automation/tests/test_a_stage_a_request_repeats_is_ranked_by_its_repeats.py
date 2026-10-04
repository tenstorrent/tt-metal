"""An op in a stage one request runs N times is ranked by N times its gap.

The full-pipeline pass runs each stage's step once, and so did the ranking's sense of cost. Qwen-Image-
Edit's denoise step runs 50 times per edit: 99% of a request's time sat in one stage, and a millisecond
saved there counted the same as one saved in a stage that runs once. The count is the pipeline's own
(<stage>_trace_repeats, pinned as KIND_STAGE_REPEATS); a stage that states nothing runs once. Stage
names here are this file's own.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parents[1]


def _led(monkeypatch, pins):
    """The REAL ledger reader (measurements.stage_repeats) over these pinned rows."""
    import cc_optimize.measurements as M

    rows = [{"kind": "stage_repeats", "depth": d, "value_ms": v} for d, v in pins.items()]
    monkeypatch.setattr(M, "rows", lambda kind="", phase="", model="", task="": [r for r in rows if r["kind"] == kind])
    return M


@pytest.fixture
def pm(monkeypatch):
    spec = importlib.util.spec_from_file_location("pm_repeats_rank_ut", str(_PA / "cc_optimize" / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "_model_key", lambda: "m")
    monkeypatch.setattr(m, "read_stage_ms", lambda **k: {})
    return m


def _profile(*stages):
    return {"stage_buckets": {s: [{"device_ms": 10.0}] for s in stages}}


def test_a_stated_count_is_read_per_stage_and_one_is_not_a_repeat(pm, monkeypatch):
    led = _led(monkeypatch, {"loop": 50.0, "once": 1.0})
    monkeypatch.setattr(pm, "_ledger", lambda: led)
    assert pm.stage_repeats_per_request(["Loop", "once", "other"]) == {"Loop": 50}, "matched on the lowercased pin"


def test_a_ledger_that_cannot_be_read_states_no_repeats(pm, monkeypatch):
    def _boom():
        raise OSError("ledger gone")

    monkeypatch.setattr(pm, "_ledger", _boom)
    assert pm.stage_repeats_per_request(["loop"]) == {}


def test_repeats_alone_weigh_the_stages(pm, monkeypatch):
    monkeypatch.setattr(pm, "_ledger", lambda: _led(monkeypatch, {"loop": 50.0}))
    assert pm.stage_cost_weights(_profile("loop", "once")) == {"loop": 50.0, "once": 1.0}


def test_repeats_multiply_the_sampling_weight(pm, monkeypatch):
    monkeypatch.setattr(pm, "_ledger", lambda: _led(monkeypatch, {"loop": 50.0}))
    monkeypatch.setattr(pm, "read_stage_ms", lambda **k: {"loop": 40.0, "once": 10.0})
    w = pm.stage_cost_weights(_profile("loop", "once"))  # sampling ratios 4 and 1 -> 1.6 and 0.4
    assert w["loop"] == pytest.approx(1.6 * 50) and w["once"] == pytest.approx(0.4)


def test_with_no_stated_count_the_weights_are_exactly_the_sampling_ones(pm, monkeypatch):
    monkeypatch.setattr(pm, "_ledger", lambda: _led(monkeypatch, {}))
    monkeypatch.setattr(pm, "read_stage_ms", lambda **k: {"loop": 40.0, "once": 10.0})
    assert pm.stage_cost_weights(_profile("loop", "once")) == pytest.approx({"loop": 1.6, "once": 0.4})
    monkeypatch.setattr(pm, "read_stage_ms", lambda **k: {})
    assert pm.stage_cost_weights(_profile("loop", "once")) == {}, "nothing to weigh by: unweighted"


def test_a_repeated_stage_s_smaller_gap_is_worked_first(pm, monkeypatch):
    monkeypatch.setattr(pm, "_ledger", lambda: _led(monkeypatch, {"loop": 50.0}))
    w = pm.stage_cost_weights(_profile("loop", "once"))
    rows = [
        {"op": "big_gap_once", "stage": "once", "eff_gap_ms": 136.0},
        {"op": "small_gap_in_loop", "stage": "loop", "eff_gap_ms": 57.0},
        {"op": "unplaced", "stage": "", "eff_gap_ms": 80.0},
    ]
    for r in rows:
        r["cost_weight"] = w.get(r["stage"], 1.0)
    order = [r["op"] for r in sorted(rows, key=lambda r: pm._blocking_order_key(r, set()))]
    assert order == ["small_gap_in_loop", "big_gap_once", "unplaced"]


def test_the_gate_s_weights_come_from_the_one_helper():
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index("def stage_cost_weights(")
    body = src[i : src.index("\ndef ", i + 1)]
    assert "stage_repeats_per_request(stages)" in body
    ms = (_PA / "cc_optimize" / "measurements.py").read_text()
    assert "def stage_repeats(" in ms
    import subprocess

    readers = subprocess.run(
        [
            "grep",
            "-rln",
            "rows(KIND_STAGE_REPEATS\\|rows(led.KIND_STAGE_REPEATS\\|KIND_STAGE_REPEATS, depth=",
            str(_PA / "cc_optimize"),
            str(_PA / "agent"),
        ],
        capture_output=True,
        text=True,
    ).stdout.split()
    assert [Path(r).name for r in readers] == ["measurements.py"], "one reader of the pinned count"


def test_emit_e2e_tells_a_model_the_ranking_depends_on_it():
    prompt = (_PA.parent.parent.parent / "scripts" / "tt_hw_planner" / "commands" / "emit_e2e.py").read_text()
    i = prompt.index("<stage>_trace_repeats(): ZERO-ARG, OPTIONAL.")
    seg = prompt[i : i + 900]
    assert "optimize weighs that stage's ops by it" in seg and "production default" in seg

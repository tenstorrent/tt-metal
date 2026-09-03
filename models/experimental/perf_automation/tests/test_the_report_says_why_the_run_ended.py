"""Limitations states why the run ended, in terms the reader can check.

A run stopped by its ROUND BUDGET with the gate still reporting can_stop=false rendered exactly like
a run the gate had cleared: same tables, same wins, no statement either way. voxtral 2026-09-03 ended
that way after 10 rounds with ~150 ms still reachable in one stack, and the report's closing section
read "6 op(s) tried but no lever beat baseline", which invites the opposite conclusion.
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
for _p in (PERF, PERF / "cc_optimize"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_spec = _ilu.spec_from_file_location("_cc_summary_why", PERF / "cc_optimize" / "summary.py")
_sm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sm)


def _log(tmp_path):
    p = tmp_path / "kl.json"
    p.write_text(
        json.dumps(
            [
                {
                    "op_signature": "MatmulDeviceOperation 512 x 3072 x 8192",
                    "kernel_kind": "knob:dtype",
                    "measured_ms": 620.0,
                    "beat_baseline": False,
                    "fullpipe_ms": 11.0,
                    "fullpipe_best_ms": 11.0,
                    "fullpipe_delta_ms": 0.1,
                    "fullpipe_measured_here": True,
                }
            ]
        )
    )
    return p


def _limits(tmp_path, facts):
    out = _sm.render_summary(_log(tmp_path), 53.25, final_override_ms=11.0, model="m", finalized=True, stop_facts=facts)
    ls = out.splitlines()
    i = ls.index("Limitations")
    return "\n".join(ls[i : i + 8])


def test_a_budget_stop_says_the_work_was_not_done(tmp_path):
    body = _limits(tmp_path, {"rounds": 10, "max_rounds": 10, "can_stop": False, "halted": False})
    assert "ROUND BUDGET" in body, body
    assert "can_stop=false" in body, "the gate's own verdict is the checkable part"
    assert "10 of 10 round(s) used" in body


def test_a_gate_cleared_run_says_so(tmp_path):
    body = _limits(tmp_path, {"rounds": 4, "max_rounds": 10, "can_stop": True, "halted": False})
    assert "can_stop=true" in body, body
    assert "ROUND BUDGET" not in body, "a cleared run must not be reported as cut off"
    assert "4 of 10 round(s) used" in body


def test_a_halted_run_is_not_reported_as_finished(tmp_path):
    body = _limits(tmp_path, {"rounds": 2, "max_rounds": 10, "can_stop": False, "halted": True})
    assert "HALTED" in body, body
    assert "ROUND BUDGET" not in body


def test_the_two_outcomes_do_not_read_alike(tmp_path):
    """The defect was that they did."""
    cut = _limits(tmp_path, {"rounds": 10, "max_rounds": 10, "can_stop": False, "halted": False})
    done = _limits(tmp_path, {"rounds": 4, "max_rounds": 10, "can_stop": True, "halted": False})
    assert cut != done


def test_a_driver_that_says_nothing_leaves_the_section_as_it_was(tmp_path):
    for facts in (None, {}, "not a dict", {"can_stop": False}):
        body = _limits(tmp_path, facts)
        assert "round(s) used" not in body, facts
        assert "no lever beat baseline" in body, "the existing findings must survive"


def test_the_line_leads_the_section(tmp_path):
    """It frames every finding under it, so it cannot sit at the bottom."""
    body = _limits(tmp_path, {"rounds": 10, "max_rounds": 10, "can_stop": False, "halted": False})
    rows = [l for l in body.splitlines() if l.startswith("- ")]
    assert rows and "ROUND BUDGET" in rows[0], rows


def test_the_driver_hands_the_facts_over(tmp_path):
    """The renderer cannot derive any of this; only the loop that stopped knows it."""
    src = (PERF / "cc_optimize" / "run.py").read_text(encoding="utf-8")
    assert '"rounds": rounds' in src and '"can_stop": can_stop' in src, "the driver must pass its own counters"
    assert "stop_facts=stop_facts" in src, "and thread them into the render"

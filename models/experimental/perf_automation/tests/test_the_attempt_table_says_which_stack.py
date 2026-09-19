"""Every attempt row names the stack its op ran in.

The table listed op, lever and two timings, and nothing said WHERE the op ran. A reader could infer
it for an op whose shape is in its name -- 32x is the token stage, 512x the prompt stage -- but the
ops with no shape (SDPA, Concat, Copy, the data movers) were unplaceable, and on voxtral those were
half of every run. Without the column a run that had spent its last twenty attempts on one stack
read exactly like one that had spread them evenly.
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

_spec = _ilu.spec_from_file_location("_cc_summary_stack", PERF / "cc_optimize" / "summary.py")
_sm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sm)


def _profile(**stages):
    return {
        "stage_buckets": {
            name: [{"top_ops": [{"op_code": c, "device_ms": ms} for c, ms in ops]}] for name, ops in stages.items()
        }
    }


def _log(tmp_path, ops):
    p = tmp_path / "kl.json"
    p.write_text(
        json.dumps(
            [
                {
                    "op_signature": op,
                    "kernel_kind": "knob:grid",
                    "measured_ms": 620.0,
                    "beat_baseline": True,
                    "fullpipe_ms": 11.0,
                    "fullpipe_best_ms": 11.1,
                    "fullpipe_delta_ms": -0.1,
                    "fullpipe_measured_here": True,
                }
                for op in ops
            ]
        )
    )
    return p


def _rows(out):
    """Header + data rows of the attempt table. The rule uses crosses, so it is not one of these."""
    ls = out.splitlines()
    i = next(i for i, l in enumerate(ls) if l.startswith("Per-attempt detail"))
    return [l for l in ls[i:] if "│" in l]


def _rule(out):
    ls = out.splitlines()
    i = next(i for i, l in enumerate(ls) if l.startswith("Per-attempt detail"))
    return next(l for l in ls[i:] if "┼" in l)


def test_the_table_has_a_stack_column(tmp_path):
    prof = _profile(warble=[("Alpha", 9.0)])
    out = _sm.render_summary(_log(tmp_path, ["Alpha"]), 53.25, model="m", finalized=True, baseline_profile=prof)
    head = _rows(out)[0]
    assert "stack" in head, head
    assert head.index("stack") < head.index("lever"), "the stack belongs beside the op, not after the lever"


def test_an_op_with_no_shape_in_its_name_is_still_placed(tmp_path):
    """The whole point: these are the rows a reader could not place by eye."""
    prof = _profile(quiet=[("Mover", 1.0)], loud=[("Mover", 30.0)])
    out = _sm.render_summary(_log(tmp_path, ["Mover"]), 53.25, model="m", finalized=True, baseline_profile=prof)
    row = next(l for l in _rows(out) if "Mover" in l)
    assert "loud" in row, row
    assert "quiet" not in row, "attributed to where it costs the most, not to every stage it touches"


def test_the_stack_name_comes_from_the_capture(tmp_path):
    """A model that calls its stacks anything is labelled in its own words."""
    prof = _profile(vocoder=[("Beta", 4.0)])
    out = _sm.render_summary(_log(tmp_path, ["Beta"]), 53.25, model="m", finalized=True, baseline_profile=prof)
    assert "vocoder" in next(l for l in _rows(out) if "Beta" in l)


def test_an_op_belonging_to_no_stack_is_blank_not_guessed(tmp_path):
    prof = _profile(alpha=[("Real", 5.0)])
    out = _sm.render_summary(_log(tmp_path, ["host_overhead"]), 53.25, model="m", finalized=True, baseline_profile=prof)
    row = next(l for l in _rows(out) if "host_overhead" in l)
    assert "alpha" not in row, "an unplaceable op must not borrow another stage's name"


def test_an_unmarked_capture_renders_the_table_unchanged(tmp_path):
    """No marks, no names -- the column is blank and nothing else moves."""
    for prof in (None, {}, {"stage_buckets": {}}):
        out = _sm.render_summary(_log(tmp_path, ["Alpha"]), 53.25, model="m", finalized=True, baseline_profile=prof)
        rows = _rows(out)
        assert "stack" in rows[0]
        assert any("Alpha" in l for l in rows), prof


def test_the_rule_is_the_one_the_target_uses(tmp_path):
    """Two answers to 'which stack is this op in' is the defect this shares a rule to avoid."""
    src = (PERF / "cc_optimize" / "summary.py").read_text(encoding="utf-8")
    assert "stage_of_op" in src, "the report must call the target's attribution rule"
    assert "def stage_of_op" not in src, "a second definition is the defect this shares a rule to avoid"
    mcp = (PERF / "cc_optimize" / "perf_mcp.py").read_text(encoding="utf-8")
    assert mcp.count("def stage_of_op") == 1, "the rule must live in exactly one place"


def test_the_divider_spans_the_widened_header(tmp_path):
    prof = _profile(alpha=[("Alpha", 5.0)])
    out = _sm.render_summary(_log(tmp_path, ["Alpha"]), 53.25, model="m", finalized=True, baseline_profile=prof)
    head, rule = _rows(out)[0], _rule(out)
    assert len(rule) >= len(head), "the rule must reach the end of the header it underlines"
    assert rule.count("┼") == head.count("│"), "every column edge needs a cross beneath it"

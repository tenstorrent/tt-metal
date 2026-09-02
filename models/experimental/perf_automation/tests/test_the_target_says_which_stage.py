"""The next target must say which stage its op lives in, or the agent falls back to the metric.

next_target named an OP and nothing else, and the only stage-shaped signal the agent gets is the
metric in its prompt -- which describes the recurring stage. Handed a stage-less op, it reasoned
about the one stage it had been told to care about and kept returning there while the ranking
pointed elsewhere. Measured on voxtral mid-run: ~5 ms of headroom left in the recurring stage and
~195 ms sitting in the prompt-consuming one, with the agent's own words naming "the largest
remaining decode lever".
"""

from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
if str(PERF) not in sys.path:
    sys.path.insert(0, str(PERF))

_spec = _ilu.spec_from_file_location("_pm_stage", PERF / "cc_optimize" / "perf_mcp.py")
_pm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pm)

_stage_of_op = _pm._stage_of_op
_stage_gap_share = _pm._stage_gap_share


def _prof(**stages):
    return {
        "stage_buckets": {
            name: [{"top_ops": [{"op_code": c, "device_ms": ms} for c, ms in ops]}] for name, ops in stages.items()
        }
    }


def test_an_op_is_attributed_to_where_it_costs_the_most_not_where_it_appears_first():
    """First-match returns whichever stage the capture serialised first -- a coin toss."""
    prof = _prof(first=[("Matmul", 1.0)], second=[("Matmul", 20.0)])
    assert _stage_of_op("Matmul", prof) == "second"


def test_a_tie_names_no_stage():
    prof = _prof(one=[("X", 5.0)], two=[("X", 5.0)])
    assert _stage_of_op("X", prof) == "", "a tie must say nothing rather than guess"


def test_an_op_the_capture_cannot_place_returns_nothing():
    """host_overhead is real and belongs to no stage; claiming one would be a lie."""
    prof = _prof(only=[("Matmul", 1.0)])
    assert _stage_of_op("host_overhead", prof) == ""
    assert _stage_of_op("", prof) == ""


def test_an_unmarked_capture_behaves_as_before_the_field_existed():
    for prof in ({}, {"stage_buckets": {}}, {"stage_buckets": None}, None):
        assert _stage_of_op("Matmul", prof) == ""
        assert _stage_gap_share(prof) == {}


def test_the_stage_names_come_from_the_capture_not_from_a_list():
    """A model that calls its stages anything must still be attributed."""
    prof = _prof(denoise=[("Conv", 9.0)], upsample=[("Conv", 2.0)])
    assert _stage_of_op("Conv", prof) == "denoise"


def test_the_share_summarises_every_marked_stage():
    prof = _prof(a=[("X", 1.5)], b=[("Y", 2.25)])
    assert _stage_gap_share(prof) == {"a": 1.5, "b": 2.25}


def test_the_target_carries_the_stage_and_the_directive_shows_where_time_is():
    src = (PERF / "cc_optimize" / "perf_mcp.py").read_text(encoding="utf-8")
    assert '"stage": _stage_of_op(blocking[0]["op"], prof)' in src
    assert "IN THE STAGE " in src, "the directive must tell the agent to work that stage"
    assert "_stage_time_note" in src, "the directive must show where the time actually is"

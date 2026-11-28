"""A roofline row headed "per request" must cover a whole request, not one step of one.

_retires_one_per_user answers "does one call retire exactly one item per user", and the heading
used that alone: anything failing it was headed "per request". A stage the pipeline replays -- it
states how many times through <stage>_trace_repeats -- fails that predicate while still being timed
one step at a time, so its row claimed a whole request over a single step's numbers.

Qwen-Image-Edit's denoise printed

    DENOISE - per request   ...   34640.09 ms

for 1 of its 50 denoise steps: 50x under what the heading said, on the stage that is ~99% of a real
edit. The headline line below the table already said so ("one request runs denoise 50x"); the table
did not, and the two contradicted each other on the same page.

The count is the pipeline's own and the stage names here are this file's.
"""

from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
if str(PERF) not in sys.path:
    sys.path.insert(0, str(PERF))

_spec = _ilu.spec_from_file_location("_summary_for_repeats_test", PERF / "cc_optimize" / "summary.py")
_summary = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_summary)

_heading_and_unit = _summary._heading_and_unit
_retires_one_per_user = _summary._retires_one_per_user

_BATCH = 32
_UNIT = "tok/s/u"
_STEP = "step"

# Retires many items per call, so it fails _retires_one_per_user -- denoise's shape.
_MANY = {"tokens": 512, "memory_ms": 1.0, "compute_ms": 2.0, "binds": "compute"}
# Retires exactly one item per user: the recurring stage the predicate was written for.
_ONE = {"tokens": _BATCH, "memory_ms": 1.0, "compute_ms": 2.0, "binds": "compute"}


def _head(stage, rf, repeats):
    return _heading_and_unit(stage, rf, _BATCH, repeats, _UNIT, _STEP)


def test_a_repeated_stage_is_headed_per_step_not_per_request():
    head, unit = _head("alpha", _MANY, {"alpha": 50})
    assert "per request" not in head, "a row covering 1 of 50 steps must not claim a whole request"
    assert head.endswith("per %s" % _STEP)
    assert unit == "%s/s" % _STEP, "its rate counts steps; items are not what the request repeats"


def test_a_stage_that_states_nothing_is_priced_exactly_as_before():
    """The seam is optional: an unstated count must leave the row untouched."""
    assert _head("alpha", _MANY, {}) == ("ALPHA — per request", "req/s")
    assert _head("alpha", _MANY, {"alpha": 1}) == ("ALPHA — per request", "req/s")
    assert _head("alpha", _MANY, None) == ("ALPHA — per request", "req/s")


def test_the_one_item_per_user_stage_keeps_its_own_unit():
    """The predicate still answers first, so the case it was written for is unchanged."""
    assert _retires_one_per_user(_ONE, _BATCH)
    assert _head("beta", _ONE, {}) == ("BETA — per %s" % _STEP, _UNIT)
    # even when that stage also states a count, the item rate is still the honest one
    assert _head("beta", _ONE, {"beta": 50}) == ("BETA — per %s" % _STEP, _UNIT)


def test_the_count_is_matched_by_the_name_the_pipeline_filed_it_under():
    """Keys are lowercased by the pin; the roof's name may be any case."""
    assert _head("Alpha", _MANY, {"alpha": 50})[0].endswith("per %s" % _STEP)
    assert _head("ALPHA", _MANY, {"alpha": 50})[0].endswith("per %s" % _STEP)
    # a count filed for a different stage must not move this row
    assert _head("alpha", _MANY, {"gamma": 50}) == ("ALPHA — per request", "req/s")


def test_a_malformed_count_does_not_cost_the_row():
    """An unreadable pin falls back to the previous heading rather than raising."""
    for bad in ({"alpha": None}, {"alpha": 0}, {"alpha": ""}):
        assert _head("alpha", _MANY, bad) == ("ALPHA — per request", "req/s")


def test_the_heading_and_the_unit_still_come_from_one_decision():
    """Both are returned by one call, so they cannot drift apart again."""
    for reps in ({}, {"alpha": 50}):
        head, unit = _head("alpha", _MANY, reps)
        per_step = head.endswith("per %s" % _STEP)
        assert per_step == (unit != "req/s"), "heading and unit disagree for repeats=%r" % (reps,)

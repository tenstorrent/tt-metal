"""The fidelity ladder names where the ceiling was PINNED and where the build runs TODAY.

One marker could only ever name one of the two, and named the pinned one. voxtral 2026-09-03 banked
7 fidelity wins that moved every stack off HiFi4, and the ladder still marked HiFi4 "in use" on all
three -- a rung the finished capture records ZERO FLOPs at.

The two facts have different sources and neither may stand in for the other:
  baseline  the peak the ceiling is ANCHORED at -- write-once, or a roof retreats ahead of the
            measurement chasing it
  now       the dominant rung of THIS capture
"""

from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
for _p in (PERF, PERF / "cc_optimize"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_spec = _ilu.spec_from_file_location("_cc_summary_ladder", PERF / "cc_optimize" / "summary.py")
_sm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sm)


def test_the_rate_label_resolves_to_the_unit_an_anchor_is_keyed_by():
    """THE SILENT MISS. Anchors are pinned at depth "token"; every caller passes "tok/s/u"."""
    assert _sm._unit_key("tok/s/u") == "token"
    assert _sm._unit_key("token") == "token"
    assert _sm._unit_key("") == "token", "no unit is not a reason to key on nothing"


def test_a_unit_that_is_not_abbreviated_survives_whole():
    """`tok` is the only abbreviation the rate builder introduces."""
    for u in ("step", "step/s", "frame/s", "request/s"):
        assert _sm._unit_key(u) == u.split("/")[0]


def test_a_peak_is_named_back_to_its_rung():
    assert _sm._rung_of_peak(175.5e12) == "hifi4"
    assert _sm._rung_of_peak(702.0e12) == "lofi"
    assert _sm._rung_of_peak(0) == ""
    assert _sm._rung_of_peak(1.0) == "", "a peak matching no rung must name none"


def test_the_observed_rung_is_read_from_the_capture(monkeypatch):
    """_peak_for_stage cannot answer this: it returns an EMPTY rung whenever a peak is pinned, so
    reading its second element gives the pinned rung on some stages and the observed one on others."""
    monkeypatch.setattr(_sm, "_fidelity_breakdown", lambda *a, **k: ([("lofi", 8e12, 702.0, 1.0)], 1.0))
    assert _sm._observed_rung("s", {"stage_buckets": {"s": [{}]}}) == "lofi"


def test_an_unmarked_capture_observes_nothing(monkeypatch):
    assert _sm._observed_rung("s", {}) == ""
    assert _sm._observed_rung("s", {"stage_buckets": {}}) == ""


def test_the_rung_list_has_one_definition():
    """A second ordering is how the ladder and the namer drift apart."""
    src = (PERF / "cc_optimize" / "summary.py").read_text(encoding="utf-8")
    assert src.count('["lofi", "hifi2", "hifi3", "hifi4"]') == 1, "the rung list is written twice"
    assert "_RUNGS" in src


def test_the_marker_never_lets_one_fact_stand_for_the_other():
    """Both markers, or the older single mark -- never a baseline label on observed data."""
    src = (PERF / "cc_optimize" / "summary.py").read_text(encoding="utf-8")
    i = src.index("def _cell(_st_c, _rf_c, _rung, _peak):")
    seg = src[i : i + 1200]
    assert "rung_baseline" not in seg, "the cell must read the resolved dicts, not re-derive"
    for _mark in ("baseline+now", "now", "baseline"):
        assert _mark in seg, _mark
    assert "in use" in seg, "a stage stating neither fact must keep the previous single mark"

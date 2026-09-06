"""A stage's compute roof is a property of the machine, not of the precision the build happens to run.

The memory roof divides by a fixed 512 GB/s, so it cannot drift and screams when the bytes do. The
compute roof was derived per render from the fidelity carrying the most FLOPs -- so the moment the
fidelity rung landed, the roof rose and the band retreated by exactly the factor the measurement had
just improved. A stage could not close a gap by lowering precision, because the gap moved with it.

Measured on voxtral_mini_3b_2507 (2026-09-05). Same capture, same bytes, only the fidelity LABEL
changed hifi4 -> lofi:

    encode   175.5 -> 702.0 TFLOPS
    prefill  175.5 -> 702.0 TFLOPS
    decode   held at 175.5 -- but only because its derivation returned nothing and it fell through
             to the pinned whole-model value. Stability by accident, on one stage.

Two faults, both here. The writer never wrote a per-stage anchor, because `flops` is derived by
roofline.annotate_op and it was handed raw buckets -- so _dominant_peak_flops answered 0.0 and the
`if _p > 0` guard skipped the write silently, for every stage of every model ever profiled. And the
reader, finding no anchor, derived one from the current build instead of falling back to the pinned
whole-model peak that was sitting right there.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
_CC = PERF / "cc_optimize"
for _p in (str(PERF), str(PERF.parent.parent.parent), str(_CC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _sm():
    from cc_optimize import summary

    return summary


def _prof(fidelity="lofi"):
    """A capture marking one stage, whose ops all run at `fidelity`."""
    op = {"op_code": "M 512 x 512 x 512", "shape": "512 x 512 x 512", "fidelity": fidelity, "device_ms": 1.0}
    return {"stage_buckets": {"s": [{"id": "matmul", "tags": {"fidelity": fidelity}, "top_ops": [op]}]}}


def test_the_roof_is_the_same_at_every_precision(monkeypatch):
    """The whole defect in one assertion: relabel the fidelity, the ceiling must not notice."""
    m = _sm()
    monkeypatch.setattr(m, "_pinned_peak_flops", lambda *a, **k: 175.5e12)
    monkeypatch.setattr(m, "_fidelity_breakdown", lambda *a, **k: ([("lofi", 8e12, 702.0, 1.0)], 1.0))
    lo = m._peak_for_stage("s", _prof("lofi"))[0]
    hi = m._peak_for_stage("s", copy.deepcopy(_prof("hifi4")))[0]
    assert lo == hi == 175.5e12


def test_a_stage_without_its_own_anchor_uses_the_pinned_model_peak(monkeypatch):
    """Falling back to a derivation is what let the roof track the build."""
    m = _sm()
    monkeypatch.setattr(m, "_pinned_peak_flops", lambda *a, **k: 175.5e12)
    monkeypatch.setattr(m, "_fidelity_breakdown", lambda *a, **k: ([("lofi", 8e12, 702.0, 1.0)], 1.0))
    peak, rung = m._peak_for_stage("s", _prof())
    assert peak == 175.5e12
    assert rung == "", "a pinned peak reports no rung; a derived one does"


def test_a_stage_anchor_still_wins_when_one_exists(monkeypatch):
    """Per-stage remains the most specific answer -- the fallback is only for its absence."""
    m = _sm()

    class _Led:
        KIND_PEAK_FLOPS = "peak_flops"

        def anchor_value(self, kind, depth="", model="", task=""):
            return 351.0e12 if depth == "s" else 175.5e12

    monkeypatch.setattr(m, "_ledger", lambda: _Led())
    assert m._peak_for_stage("s", _prof())[0] == 351.0e12


def test_nothing_pinned_at_all_still_derives(monkeypatch):
    """A model on its very first render has no anchor yet and must still print a roof."""
    m = _sm()
    monkeypatch.setattr(m, "_pinned_peak_flops", lambda *a, **k: None)
    monkeypatch.setattr(m, "_fidelity_breakdown", lambda *a, **k: ([("lofi", 8e12, 702.0, 1.0)], 1.0))
    peak, rung = m._peak_for_stage("s", _prof())
    assert peak > 0 and rung == "lofi"


def test_an_unmarked_capture_answers_nothing(monkeypatch):
    m = _sm()
    monkeypatch.setattr(m, "_pinned_peak_flops", lambda *a, **k: 175.5e12)
    assert m._peak_for_stage("s", {"stage_buckets": {}}) == (0.0, "")


def test_the_writer_prices_the_ops_before_asking_their_peak():
    """`flops` is derived, not captured. Handed raw buckets the pin computes 0 and skips, silently."""
    src = (_CC / "perf_mcp.py").read_text(encoding="utf-8")
    i = src.index("# PER-STAGE PEAKS pinned beside the per-stage bytes")
    seg = src[i : i + 1600]
    a, b = seg.index("annotate_op"), seg.index("_dominant_peak_flops")
    assert a < b, "the peak is still being taken from ops that were never priced"


def test_the_fallback_asks_for_the_key_the_anchor_is_written_under():
    """The anchor is keyed on the UNIT. Passing anything else misses, and misses silently."""
    m = _sm()
    assert m._unit_key("") == "token"
    assert m._unit_key("tok/s/u") == "token"

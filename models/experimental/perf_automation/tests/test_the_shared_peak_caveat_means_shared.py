"""The "one peak shared by every stack" caveat must mean shared, not pinned.

peak_stage holds the rung a stage DERIVED for itself, and is empty exactly when that stage's peak is
PINNED -- the comment where it is written says so outright. The caveat read it as "no stage has its
own peak". Those two questions had the same answer only while per-stage pinning was broken.

Once pinning works, every stage reports no derived rung and the caveat fires on a report whose three
stacks each hold their own anchor. Measured on voxtral_mini_3b_2507 (2026-09-06): encode, prefill and
decode each carry a peak_flops anchor, all reading 175.5 TFLOPS because the model began HiFi4
throughout, and the report still claimed there was no per-stage attribution. Agreeing on a value is
not the same as sharing one.

So the test is now whether a per-stage peak EXISTS -- derived or pinned -- which the ledger answers
directly. The caveat still fires for a model that genuinely has none, which is what it is for.
"""

from __future__ import annotations

import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
_CC = PERF / "cc_optimize"
for _p in (str(PERF), str(PERF.parent.parent.parent), str(_CC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SRC = (_CC / "summary.py").read_text(encoding="utf-8")


def _code_around(anchor: str, n: int = 1400) -> str:
    i = _SRC.index(anchor)
    seg = _SRC[i : i + n]
    return "\n".join(ln for ln in seg.splitlines() if not ln.strip().startswith("#"))


def test_a_pinned_peak_is_not_treated_as_a_missing_one():
    """The whole defect: pinned reports no rung, and no rung was read as no peak."""
    code = _code_around("def _has_own_peak")
    assert "anchor_value" in code, "the existence of a per-stage anchor is not being checked"
    assert "KIND_PEAK_FLOPS" in code


def test_a_derived_rung_still_counts_as_its_own_peak():
    """An unpinned stage that worked its own rung out has per-stage attribution too."""
    code = _code_around("def _has_own_peak")
    assert "peak_stage" in code, "the derived case must still satisfy it"


def test_the_caveat_reads_the_new_predicate():
    code = _code_around("_shared_peak = bool(_in_use)", 300)
    assert "_has_own_peak" in code
    assert "peak_stage" not in code, "the caveat is reading the derived-rung flag again"


def test_a_ledger_that_cannot_be_read_states_nothing():
    """An unreadable ledger must not assert that a stage HAS its own peak."""
    code = _code_around("def _has_own_peak")
    assert "return False" in code


def test_the_caveat_still_fires_when_nothing_is_per_stage():
    """It exists for the model that really does share one figure; that must keep working."""
    code = _code_around("_shared_peak = bool(_in_use)", 300)
    assert "not any(" in code


def test_no_stage_name_is_typed_into_the_predicate():
    code = _code_around("def _has_own_peak")
    for typed in ("decode", "prefill", "encode"):
        assert '"%s"' % typed not in code, typed

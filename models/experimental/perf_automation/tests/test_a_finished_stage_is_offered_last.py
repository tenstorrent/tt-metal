"""Knowing which stacks are still short, and choosing what to work on, were three lines apart.

_stages_short_of_achievable() is computed just above the target and used only to refuse to stop. The
target itself stayed "the biggest gap anywhere", so a stage that had reached its band kept being
handed out ahead of the one that had not.

Measured on voxtral_mini_3b_2507 (2026-09-05): encode at 77% of its peak and inside its band, decode
past its own, prefill alone still short at 136.66 ms against 26-35. Of 18 attempts, 11 went to the two
finished stacks.

Ordered, not filtered. `blocking` being empty is what makes can_stop true, so dropping entries could
end a run while those ops still had reachable rungs. Everything is still offered; a finished stage
simply stops being offered first.
"""

from __future__ import annotations

import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
_CC = PERF / "cc_optimize"
for _p in (str(PERF), str(PERF.parent.parent.parent), str(_CC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SRC = (_CC / "perf_mcp.py").read_text(encoding="utf-8")


def _order(short, rows):
    """The shipped ordering, applied to a candidate list."""
    names = {s for s in short}
    return [
        r["op"]
        for r in sorted(
            rows,
            key=lambda b: (
                1 if (names and b.get("stage") and b.get("stage") not in names) else 0,
                -(b.get("eff_gap_ms") or b.get("gap_ms") or 0.0),
            ),
        )
    ]


_ROWS = [
    {"op": "enc_big", "stage": "encode", "eff_gap_ms": 30.0},
    {"op": "dec_big", "stage": "decode", "eff_gap_ms": 26.0},
    {"op": "pre_mid", "stage": "prefill", "eff_gap_ms": 14.0},
    {"op": "unplaced", "stage": "", "eff_gap_ms": 9.0},
    {"op": "pre_small", "stage": "prefill", "eff_gap_ms": 5.0},
]


def test_the_one_stage_still_short_is_offered_first():
    """The voxtral case: a 14 ms gap in the unfinished stack beats a 30 ms gap in a finished one."""
    assert _order({"prefill"}, _ROWS)[0] == "pre_mid"


def test_finished_stages_go_to_the_back_but_are_still_offered():
    """Dropping them could empty `blocking`, and empty is what ends the run."""
    out = _order({"prefill"}, _ROWS)
    assert set(out) == {r["op"] for r in _ROWS}
    assert out.index("pre_small") < out.index("enc_big")


def test_with_nothing_short_the_order_is_untouched():
    """No bar to read means no reordering -- the run behaves exactly as before."""
    assert _order(set(), _ROWS) == ["enc_big", "dec_big", "pre_mid", "unplaced", "pre_small"]


def test_with_every_stage_short_the_order_is_untouched():
    assert _order({"encode", "prefill", "decode"}, _ROWS) == _order(set(), _ROWS)


def test_an_op_the_capture_could_not_place_keeps_its_position():
    """ "" is not a finished stage. Demoting unplaced work buries whatever the marks missed."""
    out = _order({"prefill"}, _ROWS)
    assert out.index("unplaced") < out.index("enc_big")


def test_the_stage_is_resolved_once_and_carried():
    """next_target used to recompute what the entry could have carried."""
    assert 'entry["stage"] = stage_of_op(op_code, prof)' in _SRC
    assert 'stage_of_op(blocking[0]["op"], prof)' not in _SRC, "the second resolution is back"


def test_the_bar_is_read_from_the_gate_that_owns_it():
    """One producer for 'which stacks are short' -- restating it here is how the two drift."""
    i = _SRC.index("_short_names = ")
    assert "_stages_short_of_achievable()" in _SRC[i : i + 200]


def test_no_stage_name_is_typed_into_the_ordering():
    i = _SRC.index("_short_names = ")
    seg = _SRC[i : i + 500]
    for typed in ("decode", "prefill", "encode"):
        assert '"%s"' % typed not in seg, typed

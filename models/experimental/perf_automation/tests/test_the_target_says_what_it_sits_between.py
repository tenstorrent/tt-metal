"""The target names the op's neighbours, so a chain can be reasoned about.

tracy_tool._neighbours reads execution order out of the capture -- its docstring says it exists
because "nothing downstream could ask what does this op feed into" -- and roofline carries prev_op /
next_op onto every open_op row. The blocking entry then copied five fields and dropped both, so the
one thing the agent is handed described an op standing alone.

That is the shape of what is left on voxtral 2026-09-04: 13,580 data-movement ops, 12% of the run,
each existing because a producer emits a layout its consumer cannot use. A reshard is a property of
an EDGE. An agent shown only nodes cannot see one.
"""

from __future__ import annotations

import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
if str(PERF) not in sys.path:
    sys.path.insert(0, str(PERF))

_SRC = (PERF / "cc_optimize" / "perf_mcp.py").read_text(encoding="utf-8")

# black reflows a long concatenation across lines, so an exact-text read of one cannot see it.
# Collapsing runs of whitespace keeps token ORDER, which is what these last checks are about.
_FLAT = " ".join(_SRC.split())


def _slice(after: str, n: int = 1400) -> str:
    return _SRC[_SRC.index(after) : _SRC.index(after) + n]


def test_the_blocking_entry_keeps_the_adjacency_roofline_carried():
    seg = _slice('op_code = o.get("op_code") or o.get("bucket") or ""')
    assert '"prev_op": o.get("prev_op")' in seg
    assert '"next_op": o.get("next_op")' in seg


def test_the_target_handed_to_the_agent_carries_it():
    seg = _slice("next_target = (")
    assert '"prev_op": blocking[0].get("prev_op")' in seg
    assert '"next_op": blocking[0].get("next_op")' in seg


def test_an_op_the_capture_cannot_place_reports_empty_not_missing():
    """Absent must read as "" -- a missing key makes the caller guess, which is the failure the
    stage field already had."""
    seg = _slice("next_target = (")
    assert '"prev_op": blocking[0].get("prev_op") or ""' in seg
    assert '"next_op": blocking[0].get("next_op") or ""' in seg


def test_the_directive_says_what_the_op_sits_between():
    i = _SRC.index('_chain_note = ""')
    seg = _SRC[i : i + 600]
    assert "runs between" in seg
    assert "belongs to the pair, not to the op alone" in seg, "the point is the join, not the neighbours"


def test_the_note_is_silent_when_there_is_no_adjacency():
    """A capture that recorded no neighbours must add nothing to the directive."""
    i = _SRC.index('_chain_note = ""')
    seg = _SRC[i : i + 300]
    assert 'if next_target and (next_target.get("prev_op") or next_target.get("next_op")):' in seg


def test_the_note_reaches_the_directive():
    assert "+ _chain_note +" in _FLAT, "built but never used is the defect this fixes one level up"


def test_the_note_is_built_after_the_target_it_reads():
    a = _FLAT.index("next_target = (")
    b = _FLAT.index('_chain_note = ""')
    c = _FLAT.index("+ _chain_note +")
    assert a < b < c


def test_no_op_name_is_typed_here():
    """Neighbours come from the capture; a name written here would survive the model that had it."""
    i = _SRC.index('_chain_note = ""')
    seg = _SRC[i : i + 600]
    for typed in ("Matmul", "Concat", "Reshard", "SDPA", "LayerNorm"):
        assert typed not in seg, typed

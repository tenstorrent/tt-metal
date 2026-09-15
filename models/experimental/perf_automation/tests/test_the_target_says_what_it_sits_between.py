"""The target names the op's neighbours, so a chain can be reasoned about.

tracy_tool._neighbours reads execution order out of the capture and roofline carries prev_op /
next_op onto every open_op row. The blocking entry copied five fields and dropped both, so the one
thing the agent is handed described an op standing alone. These carry them the last hop.

The DIRECTIVE deliberately says nothing about them. A sentence telling the agent that cost belongs
to a join fired on every target, and on this model the repeated chains are all in decode -- the one
stage already at its floor. Steering every target toward pair-thinking to serve a finished stage
costs the stages that are still open. The fields are data; the agent decides what they mean.
"""

from __future__ import annotations

import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
if str(PERF) not in sys.path:
    sys.path.insert(0, str(PERF))

_SRC = (PERF / "cc_optimize" / "perf_mcp.py").read_text(encoding="utf-8")


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

"""Each round announces which stacks still owe their band before it starts.

The loop already ENFORCED the condition -- it re-checks can_stop and starts another round -- but
nothing said why, so an agent that wrapped up early looked exactly like one that had finished. That
is how voxtral 2026-09-03 spent ten rounds each ending with a tidy summary while can_stop was false
and ~150 ms sat reachable in one stack.
"""

from __future__ import annotations

import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
if str(PERF) not in sys.path:
    sys.path.insert(0, str(PERF))

_SRC = (PERF / "cc_optimize" / "run.py").read_text(encoding="utf-8")


def test_the_gate_probe_reports_the_shortfall():
    """_gate_status asks the gate directly; the shortfall must come back with can_stop."""
    i = _SRC.index("def _gate_status(")
    seg = _SRC[i : i + 2600]
    assert "stages_short_of_achievable" in seg, "the probe must read the gate's own field"
    assert "SHORT=" in seg, "and carry it back over the subprocess boundary"


def test_the_shortfall_survives_parsing_and_reaches_the_caller():
    i = _SRC.index("def _gate_status(")
    seg = _SRC[i : i + 3200]
    assert 'line.startswith("SHORT=")' in seg, "the reply must be parsed"
    assert '"short": short' in seg, "and returned"


def test_an_unreadable_gate_returns_the_field_too():
    """The early return must carry the same shape, or the caller reads a missing key."""
    i = _SRC.index("def _gate_status(")
    seg = _SRC[i : i + 3200]
    j = seg.index("if rc is None:")
    assert '"short": ""' in seg[j : j + 200], "the failure path must not drop the field"


def test_the_loop_prints_it_before_starting_a_round():
    i = _SRC.index("while rounds < max_rounds:")
    seg = _SRC[i : i + 2000]
    k = seg.index("_run_round_with_watchdog(round_cmd")
    before = seg[:k]
    assert 'st.get("short")' in before, "the round must announce its target before it runs"
    assert "still short of their band" in before


def test_it_is_printed_only_when_something_is_short():
    """A run with every stack inside its band must not print an empty line every round."""
    i = _SRC.index("while rounds < max_rounds:")
    seg = _SRC[i : i + 2000]
    j = seg.index('st.get("short")')
    assert seg[max(0, j - 40) : j].rstrip().endswith("if"), "the print must be guarded by the field"

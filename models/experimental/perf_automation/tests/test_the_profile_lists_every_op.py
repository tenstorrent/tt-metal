"""The profile records every distinct op, ranked by hotness. Ranking is priority, not a cut.

`_top_ops(members, available_cores, k: int = 6)` grouped a bucket's ops by (op, shape, memory), sorted
by device_ms, and returned `out[:k]`. Everything past the sixth fingerprint in each bucket was folded
into the bucket total and never appeared as an op again -- not in the roofline, not in open_ops, not
in blocking_ops, not in the report's matrix.

That is the same mistake as the two caps in roofline.py and perf_mcp.py, one layer further upstream:
a display limit standing in for a work queue. Removing those two widened the queue, but the queue is
fed from this list, so an op cut here could still never be selected.

gemma-3-12b-it is 8 buckets, so at k=6 the whole model was describable in at most 48 op fingerprints
regardless of how many it actually runs.

Hotness still decides ORDER -- the sort is unchanged and the gate works the largest gap first. It no
longer decides EXISTENCE. PERF_MCP_TOP_OPS_MAX caps the list for anyone who needs the old bounded
size, and is unset by default.
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from models.experimental.perf_automation.agent import tracy_tool as T  # noqa: E402


def _members(n: int):
    """n distinct op fingerprints in one bucket, descending device time."""
    out = []
    for i in range(n):
        out.append(
            {
                "report": {"OP Code": "Op%02d" % i, "Device Time": (n - i) * 1000.0, "Cores": 8},
                "raw": {"INPUT_0_MEMORY": "DEV_0_DRAM_INTERLEAVED", "MATH FIDELITY": "LoFi"},
            }
        )
    return out


# ---------------------------------------------------------------- the cap


def test_every_distinct_op_is_listed(monkeypatch):
    """20 fingerprints in, 20 out -- not 6."""
    monkeypatch.delenv("PERF_MCP_TOP_OPS_MAX", raising=False)
    assert len(T._top_ops(_members(20), 110)) == 20


def test_the_op_that_used_to_be_cut_is_present(monkeypatch):
    """The 7th-hottest op was unreachable at k=6. PagedUpdateCache sat in this position."""
    monkeypatch.delenv("PERF_MCP_TOP_OPS_MAX", raising=False)
    codes = [o["op_code"] for o in T._top_ops(_members(20), 110)]
    assert "Op06" in codes and "Op19" in codes


def test_hotness_still_decides_order(monkeypatch):
    """Priority is the whole point of the ranking and must survive."""
    monkeypatch.delenv("PERF_MCP_TOP_OPS_MAX", raising=False)
    ms = [o["device_ms"] for o in T._top_ops(_members(12), 110)]
    assert ms == sorted(ms, reverse=True), ms


def test_a_bucket_with_few_ops_is_unchanged(monkeypatch):
    monkeypatch.delenv("PERF_MCP_TOP_OPS_MAX", raising=False)
    assert len(T._top_ops(_members(3), 110)) == 3


def test_an_empty_bucket_is_empty(monkeypatch):
    monkeypatch.delenv("PERF_MCP_TOP_OPS_MAX", raising=False)
    assert T._top_ops([], 110) == []


# ---------------------------------------------------------------- the opt-in bound


def test_an_explicit_cap_is_honoured(monkeypatch):
    """Anyone who needs a bounded profile can ask for one -- but it is a decision someone makes,
    not a constant nobody remembers choosing."""
    monkeypatch.setenv("PERF_MCP_TOP_OPS_MAX", "5")
    assert len(T._top_ops(_members(20), 110)) == 5


def test_a_junk_cap_is_ignored(monkeypatch):
    for bad in ("0", "-3", "abc", ""):
        monkeypatch.setenv("PERF_MCP_TOP_OPS_MAX", bad)
        assert len(T._top_ops(_members(9), 110)) == 9, bad


def test_the_k_argument_still_works_when_passed(monkeypatch):
    """Callers that pass k explicitly keep their behaviour."""
    monkeypatch.delenv("PERF_MCP_TOP_OPS_MAX", raising=False)
    assert len(T._top_ops(_members(20), 110, k=4)) == 4


# ---------------------------------------------------------------- grouping unchanged


def test_identical_fingerprints_still_group(monkeypatch):
    """The list grows because more DISTINCT ops survive, not because grouping broke."""
    monkeypatch.delenv("PERF_MCP_TOP_OPS_MAX", raising=False)
    dup = _members(1) * 5
    out = T._top_ops(dup, 110)
    assert len(out) == 1 and out[0]["count"] == 5

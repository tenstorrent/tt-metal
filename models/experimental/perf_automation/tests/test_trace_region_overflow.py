import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cc_optimize import perf_mcp as m


def test_dram_trace_sig_true():
    assert m._dram_trace_sig("trace region is full; grow the trace") is True
    assert m._dram_trace_sig("trace_region_size too small") is True
    assert m._dram_trace_sig("ran out of space in trace buffer") is True


def test_dram_trace_sig_false():
    assert m._dram_trace_sig("could not trace at all (path=None)") is False
    assert m._dram_trace_sig("trace region reserved ok") is False


def test_dram_does_not_match_l1_and_vice_versa():
    l1 = "Circular buffer grow to beyond max L1 size of 1.5MB"
    dram = "trace region is full, grow the trace"
    assert m._dram_trace_sig(l1) is False
    assert m._is_l1_overflow(dram) is False
    assert m._is_l1_overflow(l1) is True
    assert m._dram_trace_sig(dram) is True


def test_is_dram_trace_overflow_inline():
    assert m._is_dram_trace_overflow("trace region full — grow the trace") is True
    assert m._is_dram_trace_overflow("generic crash") is False


def test_grow_trace_region_doubles_and_caps():
    saved = os.environ.get("TT_PERF_TRACE_REGION")
    try:
        os.environ.pop("TT_PERF_TRACE_REGION", None)
        n1 = m._grow_trace_region()
        assert n1 == m._TRACE_REGION_DEFAULT * 2
        n2 = m._grow_trace_region()
        assert n2 == n1 * 2
        os.environ["TT_PERF_TRACE_REGION"] = str(m._TRACE_REGION_MAX)
        capped = m._grow_trace_region()
        assert capped == m._TRACE_REGION_MAX
    finally:
        if saved is None:
            os.environ.pop("TT_PERF_TRACE_REGION", None)
        else:
            os.environ["TT_PERF_TRACE_REGION"] = saved


def test_dram_overflow_msg_content():
    msg = m._dram_overflow_msg(95551488)
    assert "DRAM_TRACE_OVERFLOW" in msg
    assert "95551488" in msg
    assert "TT_PERF_TRACE_REGION" in msg

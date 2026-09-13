"""KV-cache decode gate — measurement-gated, un-conflated from the trace-capture lever (no hardware).

Covers the fix that stops the optimize agent from dismissing a repeat_prefill decode as
'irreducible' after applying the trace-capture lever: the kv-cache lever is a SEPARATE gate that
clears only on a measured per-token reduction (bounded retries), and the host/dispatch ladder no
longer declares the residual blanket-irreducible.
"""

import sys
from pathlib import Path

_CC = Path(__file__).resolve().parents[1] / "cc_optimize"
if str(_CC) not in sys.path:
    sys.path.insert(0, str(_CC))

import perf_mcp  # noqa: E402


def _prof(status="repeat_prefill", host_ms=200.0, per_token=12.0):
    return {
        "decode_status": status,
        "per_token_ms": per_token,
        "buckets": [{"id": "host_overhead", "device_ms": host_ms}],
    }


def _kv(beat, kind="kv-cache", measured_ms=8.0):
    """A win carries the MEASUREMENT that makes it one -- record_kernel_attempt takes measured_ms as a
    required argument, so a real winning row always has it. This fixture used to omit it, which let
    the gate be tested against a row production cannot produce, and hid that the gate cleared on the
    flag alone despite its docstring promising 'ONLY on a MEASURED per-token reduction'."""
    return {"kernel_kind": kind, "beat_baseline": beat, "measured_ms": measured_ms}


def test_decode_gate_fires_on_repeat_prefill_with_no_attempts():
    g = perf_mcp._decode_gate(_prof(), [])
    assert g is not None
    assert g["op"] == "generation_loop"
    assert g["next_rung"] == "structural-decode"
    assert "MANDATORY" in g["reason"] and "kv-cache" in g["reason"].lower()


def test_trace_structural_attempt_does_NOT_clear_kv_gate():
    # The exact failure we hit: agent records a generic 'structural'/trace attempt.
    attempts = [{"kernel_kind": "structural", "beat_baseline": False}]
    g = perf_mcp._decode_gate(_prof(), attempts)
    assert g is not None, "a trace/structural attempt must NOT satisfy the kv-cache gate"


def test_failed_kv_attempts_keep_blocking_until_cap():
    attempts = [_kv(False), _kv(False)]  # 2 real kv-cache attempts, none won
    g = perf_mcp._decode_gate(_prof(), attempts)
    assert g is not None, "below the retry cap, unproven kv-cache keeps blocking"


def test_winning_kv_attempt_clears_gate():
    attempts = [_kv(False), _kv(True)]  # measured per-token reduction
    assert perf_mcp._decode_gate(_prof(), attempts) is None


def test_retry_cap_yields_to_avoid_infinite_loop(monkeypatch):
    monkeypatch.setenv("PERF_MCP_MAX_KV_ATTEMPTS", "3")
    attempts = [_kv(False), _kv(False), _kv(False)]  # cap reached, none won
    assert perf_mcp._decode_gate(_prof(), attempts) is None


def test_no_gate_when_decode_is_not_recompute():
    # traced single-token decode already present, and no capacity scaling -> no kv-cache demand
    assert perf_mcp._decode_gate(_prof(status="traced"), []) is None


def test_host_ladder_asks_for_trace_not_structural_and_avoids_irreducible():
    host_op = {"bound_by": "host", "bucket": "host_fallback", "grid": "", "weight_dtype": ""}
    done, rung, reason = perf_mcp._op_ladder_status(host_op, "host_overhead", [])
    assert not done and rung == "trace-capture"
    # One failed attempt no longer closes the rung -- see test_the_host_rung_clears_on_a_measurement.
    # It retires on a measured win or at the attempt cap, and the surviving intent of this test is
    # what the reason says WHEN it retires: never a blanket "irreducible" (the word the agent
    # parroted), but a hand-off to the KV-cache lever.
    _tried = [
        {"kernel_kind": k, "op_signature": "host_overhead", "beat_baseline": False, "measured_ms": 400.0}
        for k in ("trace-capture", "trace", "structural")
    ]
    done_once, _, _ = perf_mcp._op_ladder_status(host_op, "host_overhead", _tried[:1])
    assert not done_once, "a single measured-no-gain attempt must not seal the dispatch axis"
    done2, _, reason2 = perf_mcp._op_ladder_status(host_op, "host_overhead", _tried)
    assert done2
    assert "kv-cache" in reason2.lower()
    assert "not irreducible" in reason2.lower()


# --- wedge tolerance: a KV-cache attempt that WEDGES the device must count as "tried" ------------
# A wedged kv-cache attempt is auto-recorded with kernel_detected_in_source=False, so it is dropped
# from the `attempts` the gate is handed; the gate counts such wedges from the full attempt log so a
# KV-cache that crashes every time yields (like a wedged tt-lang/C++ kernel) instead of looping.


def _wedge(kind="structural-decode"):
    return {"kernel_kind": kind, "beat_baseline": False, "wedged": True, "kernel_detected_in_source": False}


def test_wedged_kv_attempts_count_toward_cap(monkeypatch):
    monkeypatch.setenv("PERF_MCP_MAX_KV_ATTEMPTS", "3")
    # 3 device wedges, none surfaced in the (detected-filtered) `attempts` list
    monkeypatch.setattr(perf_mcp, "_load_attempts", lambda: [_wedge(), _wedge(), _wedge()])
    assert perf_mcp._decode_gate(_prof(), []) is None, "3 wedged kv-cache attempts must retire the gate"


def test_below_cap_wedges_still_block(monkeypatch):
    monkeypatch.setenv("PERF_MCP_MAX_KV_ATTEMPTS", "3")
    monkeypatch.setattr(perf_mcp, "_load_attempts", lambda: [_wedge(), _wedge()])
    assert perf_mcp._decode_gate(_prof(), []) is not None, "below the cap, wedges keep blocking"


def test_mixed_clean_and_wedged_reach_cap(monkeypatch):
    monkeypatch.setenv("PERF_MCP_MAX_KV_ATTEMPTS", "3")
    # 1 clean (surfaced) + 2 wedged (from the log) == cap
    monkeypatch.setattr(perf_mcp, "_load_attempts", lambda: [_wedge(), _wedge()])
    assert perf_mcp._decode_gate(_prof(), [_kv(False)]) is None


def test_wedge_of_a_different_rung_does_not_count(monkeypatch):
    monkeypatch.setenv("PERF_MCP_MAX_KV_ATTEMPTS", "3")
    # a matmul grid wedge is NOT a kv-cache attempt and must not advance the kv cap
    monkeypatch.setattr(perf_mcp, "_load_attempts", lambda: [_wedge("grid"), _wedge("grid"), _wedge("grid")])
    assert perf_mcp._decode_gate(_prof(), []) is not None


def test_clean_win_clears_even_with_wedges(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_load_attempts", lambda: [_wedge()])
    assert perf_mcp._decode_gate(_prof(), [_kv(True)]) is None, "a measured win still clears immediately"


def test_an_unmeasured_win_row_does_NOT_clear_the_gate():
    """The gate's contract in its own words: it clears ONLY on a MEASURED per-token reduction. A
    legacy row flagged as a win with nothing timed must not release the run from KV-cache work."""
    attempts = [{"kernel_kind": "kv-cache", "beat_baseline": True, "measured_ms": None}]
    assert perf_mcp._decode_gate(_prof(), attempts) is not None


def test_the_gate_and_the_report_agree_about_one_row():
    """Same row, same verdict, both sides -- the gate deciding differently from the report is how the
    run acted on a win the report refused to show."""
    import importlib.util as _ilu
    from pathlib import Path as _P

    _spec = _ilu.spec_from_file_location("sm_kv_ut", _P(__file__).resolve().parents[1] / "cc_optimize" / "summary.py")
    _sm = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_sm)
    for row, expect_win in (
        (_kv(True), True),
        (_kv(False), False),
        ({"kernel_kind": "kv-cache", "beat_baseline": True}, False),
    ):
        gate_cleared = perf_mcp._decode_gate(_prof(), [row]) is None
        assert gate_cleared == _sm._is_win(row) == expect_win, row

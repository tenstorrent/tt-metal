"""KV-cache decode gate — measurement-gated, un-conflated from trace/2CQ (no hardware).

Covers the fix that stops the optimize agent from dismissing a repeat_prefill decode as
'irreducible' after applying trace/2CQ: the kv-cache lever is a SEPARATE gate that clears
only on a measured per-token reduction (bounded retries), and the host/dispatch ladder no
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


def _kv(beat, kind="kv-cache"):
    return {"kernel_kind": kind, "beat_baseline": beat}


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
    assert not done and rung == "trace-2cq"
    # tried state must NOT blanket-declare irreducible (that was the word the agent parroted)
    done2, _, reason2 = perf_mcp._op_ladder_status(
        host_op,
        "host_overhead",
        [{"kernel_kind": "trace-2cq", "op_signature": "host_overhead", "beat_baseline": False}],
    )
    assert done2
    assert "kv-cache" in reason2.lower()

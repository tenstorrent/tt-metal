# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generalized (physics-based) measurement-integrity guard: _comparable should reject a crashed/
partial capture by the model's OWN roofline floor (model-agnostic), NOT by op-count drop (which
false-positived on legitimate op-reducing fusions like MoE expert batching). Op-count INFLATION
stays a hard reject; with no floor it falls back to the old op-count heuristic."""

from agent.handlers.remeasure import _comparable


def _prof(device_ms, matmul_count):
    return {"device_ms": device_ms, "buckets": [{"id": "matmul", "count": matmul_count, "device_ms": device_ms}]}


BASE = _prof(40.0, 4026)  # nemotron-ish baseline; roofline floor ~15.3


def test_below_floor_is_rejected_crashed_capture():
    # seamless-style fake: 0.46ms is far below the floor -> physically impossible -> reject.
    fake = _prof(0.46, 5)
    ok, reason = _comparable(BASE, fake, floor_ms=15.3)
    assert ok is False and "below_roofline_floor" in reason


def test_op_count_drop_above_floor_is_accepted_fusion():
    # nemotron MoE batching: op-count collapses (4026 -> 200) BUT device_ms 34.5 is well above the
    # floor -> physically plausible -> a LEGITIMATE fusion. The old guard wrongly rejected this.
    batched = _prof(34.5, 200)
    ok, reason = _comparable(BASE, batched, floor_ms=15.3)
    assert ok is True and reason is None


def test_knob_win_accepted():
    # ordinary knob win: op-count unchanged, device_ms above floor -> accept.
    win = _prof(20.0, 4026)
    ok, reason = _comparable(BASE, win, floor_ms=15.3)
    assert ok is True and reason is None


def test_inflation_still_rejected_regardless_of_floor():
    inflated = _prof(40.0, 6000)  # 1.49x baseline ops -> measurement double-count
    ok, reason = _comparable(BASE, inflated, floor_ms=15.3)
    assert ok is False and "op_count_inflated" in reason


def test_backward_compat_no_floor_uses_op_count_heuristic():
    # with no roofline floor, the old conservative op-count-drop heuristic still guards.
    dropped = _prof(0.46, 5)
    ok, reason = _comparable(BASE, dropped, floor_ms=None)
    assert ok is False and "structural_op_dropped" in reason


def _profw(device_ms, matmul_count, forward_wall_ms):
    p = _prof(device_ms, matmul_count)
    p["forward_wall_ms"] = forward_wall_ms
    return p


BASEW = _profw(148.0, 4026, 300.0)  # seamless-ish complete baseline (above floor)


def test_partial_capture_above_floor_rejected_by_forward_wall():
    partial = _profw(74.0, 3400, 300.0)
    ok, reason = _comparable(BASEW, partial, floor_ms=15.3)
    assert ok is False and "capture_incomplete" in reason


def test_real_speedup_drops_both_device_and_wall_accepted():
    win = _profw(100.0, 4026, 210.0)
    ok, reason = _comparable(BASEW, win, floor_ms=15.3)
    assert ok is True and reason is None


def test_forward_wall_absent_degrades_gracefully():
    partial = _prof(74.0, 3400)
    ok, reason = _comparable(BASE, partial, floor_ms=15.3)
    assert ok is True and reason is None


def _profpt(device_ms, matmul_count, forward_wall_ms, per_token_ms):
    p = _profw(device_ms, matmul_count, forward_wall_ms)
    p["per_token_ms"] = per_token_ms
    return p


BASEPT = _profpt(28.72, 4026, 300.0, 5.0)


def test_per_token_preferred_host_bound_win_accepted():
    win = _profpt(10.62, 4026, 299.0, 2.2)
    ok, reason = _comparable(BASEPT, win, floor_ms=8.0)
    assert ok is True and reason is None


def test_per_token_flat_partial_capture_still_rejected():
    partial = _profpt(10.62, 3400, 299.0, 4.98)
    ok, reason = _comparable(BASEPT, partial, floor_ms=8.0)
    assert ok is False and "capture_incomplete" in reason and "trace per-token" in reason


def test_forward_wall_fallback_when_per_token_absent():
    partial = _profw(74.0, 3400, 300.0)
    ok, reason = _comparable(BASEW, partial, floor_ms=15.3)
    assert ok is False and "capture_incomplete" in reason and "forward wall" in reason

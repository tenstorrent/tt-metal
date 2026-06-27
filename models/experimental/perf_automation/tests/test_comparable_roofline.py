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

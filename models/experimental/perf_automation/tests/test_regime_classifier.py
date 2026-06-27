# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""knob-vs-kernel regime classifier: roofline bound_by + tried-levers -> verdict."""

from agent import roofline, states

KL = states.KERNEL_LEVER
FP = states.FROM_PRINCIPLES
KNOBS = ["qkv-program-config", "shard-activation-to-l1"]
CANDS = KNOBS + [KL, FP]


def _bucket(bound_by, ideal_ms=1.0, gap_ms=5.0):
    return {
        "id": "matmul",
        "top_ops": [
            {"op_code": "matmul", "device_ms": 6.0, "ideal_ms": ideal_ms, "gap_ms": gap_ms, "bound_by": bound_by}
        ],
    }


def test_dispatch_bound_is_kernel_from_iter0():
    # launch-bound dominant op -> kernel immediately, no knobs tried yet
    r = roofline.classify_regime(
        _bucket("dispatch"), tried=[], candidate_ids=CANDS, kernel_lever=KL, from_principles=FP
    )
    assert r["verdict"] == "kernel"
    assert "dispatch" in r["why"] or "launch" in r["why"]


def test_compute_bound_with_untried_knobs_is_knob():
    r = roofline.classify_regime(_bucket("compute"), tried=[], candidate_ids=CANDS, kernel_lever=KL, from_principles=FP)
    assert r["verdict"] == "knob"
    assert "grid/fidelity" in r["why"]


def test_memory_bound_with_untried_knobs_is_knob():
    r = roofline.classify_regime(_bucket("memory"), tried=[], candidate_ids=CANDS, kernel_lever=KL, from_principles=FP)
    assert r["verdict"] == "knob"
    assert "dtype/shard" in r["why"]


def test_all_knobs_tried_escalates_to_kernel():
    # compute-bound but every knob already tried -> structural residual -> kernel
    r = roofline.classify_regime(
        _bucket("compute"), tried=KNOBS, candidate_ids=CANDS, kernel_lever=KL, from_principles=FP
    )
    assert r["verdict"] == "kernel"
    assert r["knobs_remaining"] == []


def test_at_floor_with_knobs_spent_is_kernel():
    # dominant op within 10% of its floor and knobs spent -> between-op cost -> kernel
    b = _bucket("compute", ideal_ms=5.0, gap_ms=0.2)  # gap <= 10% of ideal -> at_floor
    r = roofline.classify_regime(b, tried=KNOBS, candidate_ids=CANDS, kernel_lever=KL, from_principles=FP)
    assert r["verdict"] == "kernel"
    assert r["at_floor"] is True


def test_kernel_verdict_degrades_when_ttl_unavailable():
    # dispatch-bound would be 'kernel', but if ttl isn't available it must not emit an unactionable verdict
    r = roofline.classify_regime(
        _bucket("dispatch"),
        tried=[],
        candidate_ids=KNOBS + [FP],
        kernel_lever=KL,
        from_principles=FP,
        kernel_available=False,
    )
    assert r["verdict"] == "knob"  # falls back to an untried knob
    assert "fall back" in r["why"]


def test_no_signal_prefers_untried_knob():
    r = roofline.classify_regime(
        _bucket(None, ideal_ms=None, gap_ms=None), tried=[], candidate_ids=CANDS, kernel_lever=KL, from_principles=FP
    )
    assert r["verdict"] == "knob"

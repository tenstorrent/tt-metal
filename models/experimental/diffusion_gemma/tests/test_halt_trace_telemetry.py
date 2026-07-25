# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Halt-trace telemetry: does the emitted summary identify WHICH gate blocked? (#48291)

The traced controller computes ``(steps_run, mean_entropy, mismatch)`` every denoise step and
used to keep only the ``halted`` boolean. That boolean cannot distinguish the three ways a
block can burn all 48 steps, and they need opposite fixes:

* the entropy floors structurally above the bar (content never converged),
* the entropy misses the bar by a numerical hair,
* the entropy clears the bar but the argmax never stops moving.

These are CPU-only, checkpoint-free, device-free tests of that reduction.
"""

import pytest

from models.experimental.diffusion_gemma.tt.traced_denoise import _summarize_halt_trace

THRESHOLD = 0.005


def _trace(pairs):
    """Build a ``last_halt_trace`` from ``[(mean_entropy, mismatch), ...]`` in step order."""
    return [(index + 1, entropy, mismatch) for index, (entropy, mismatch) in enumerate(pairs)]


def test_halted_block_reports_no_blocking_gate():
    summary = _summarize_halt_trace(_trace([(0.9, 4.0), (0.004, 0.0)]), threshold=THRESHOLD)
    assert summary["halt_blocking_gate"] == "none"
    assert summary["halt_steps_both_gates"] == 1


def test_structural_entropy_floor_is_named_and_scaled():
    """The 0.14-0.51 nats regime: argmax goes stable, entropy is 30-100x the bar."""
    summary = _summarize_halt_trace(_trace([(0.6, 9.0)] + [(0.5, 0.0)] * 47), threshold=THRESHOLD)
    assert summary["halt_blocking_gate"] == "entropy"
    assert summary["halt_steps_mismatch_zero"] == 47
    assert summary["halt_steps_entropy_under_threshold"] == 0
    # The ratio is what separates this from a near-miss; 0.5 / 0.005 = 100x.
    assert summary["halt_entropy_floor_ratio"] == pytest.approx(100.0, rel=1e-3)
    assert summary["halt_entropy_margin_final"] > 0.4


def test_numerical_near_miss_is_distinguishable_from_the_floor():
    """Same blocking gate as the floor case, but the margin/ratio must expose the difference."""
    summary = _summarize_halt_trace(_trace([(0.6, 9.0)] + [(0.0051, 0.0)] * 47), threshold=THRESHOLD)
    assert summary["halt_blocking_gate"] == "entropy"
    assert summary["halt_entropy_floor_ratio"] == pytest.approx(1.02, rel=1e-2)
    assert 0.0 < summary["halt_entropy_margin_final"] < 0.001


def test_oscillating_argmax_is_attributed_to_the_mismatch_gate():
    summary = _summarize_halt_trace(_trace([(0.6, 9.0)] + [(0.001, 3.0)] * 47), threshold=THRESHOLD)
    assert summary["halt_blocking_gate"] == "mismatch"
    assert summary["halt_steps_entropy_under_threshold"] == 47
    assert summary["halt_steps_mismatch_zero"] == 0
    assert summary["halt_mismatch_final"] == 3.0


def test_neither_gate_ever_satisfied_reports_both():
    summary = _summarize_halt_trace(_trace([(0.5, 7.0)] * 48), threshold=THRESHOLD)
    assert summary["halt_blocking_gate"] == "both"


def test_gates_satisfied_on_different_steps_is_not_a_halt():
    """Both gates pass, never on the same step -- must not read as ``none`` (that would imply
    eval_halt should have fired) nor as a single-gate failure."""
    summary = _summarize_halt_trace(_trace([(0.5, 9.0), (0.001, 4.0), (0.4, 0.0), (0.001, 2.0)]), threshold=THRESHOLD)
    assert summary["halt_blocking_gate"] == "never_simultaneous"
    assert summary["halt_steps_both_gates"] == 0


def test_step_zero_pass_is_not_counted_as_eligible():
    """eval_halt requires ``step >= 1``, so a step-0-only pass is NOT a missed halt."""
    summary = _summarize_halt_trace(_trace([(0.001, 0.0), (0.5, 6.0), (0.5, 6.0)]), threshold=THRESHOLD)
    assert summary["halt_eligible_steps"] == 2
    assert summary["halt_steps_both_gates"] == 0
    assert summary["halt_blocking_gate"] == "both"


def test_empty_trace_is_reported_not_crashed():
    summary = _summarize_halt_trace([], threshold=THRESHOLD)
    assert summary["halt_trace_steps"] == 0
    assert summary["halt_blocking_gate"] == "none"


def test_summary_is_json_serializable_for_the_metric_channel():
    import json

    summary = _summarize_halt_trace(_trace([(0.5, 7.0)] * 3), threshold=THRESHOLD)
    assert json.loads(json.dumps(summary))["halt_trace_steps"] == 3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The perf gate: a two-sided band per mode (untraced ±5%, traced ±3%). Faster than the band fails too ("re-centre
it"). A mode without a target only reports."""

import pytest

from models.common.ci_failure import assert_perf_gate

MARGIN = {"untraced": 0.05, "traced": 0.03}


def band_failure(what, measured, target, margin, unit="ms"):
    """Failure text if `measured` is outside target ± margin, else None (also None without a target)."""
    if target is None:
        return None
    lower, upper = target * (1 - margin), target * (1 + margin)
    if lower <= measured <= upper:
        return None
    direction = "faster than the target, re-centre it" if measured < lower else "slower than the target"
    return (
        f"{what}: {measured:.3f} {unit} outside [{lower:.3f}, {upper:.3f}] "
        f"(target {target:.3f} {unit} ±{margin:.0%}); {direction}"
    )


def gate(label, results, correctness=(), skip_unmeasured=True):
    """`results`: {mode: (samples_ms, measured_ms, target_ms or None, margin)}. Prints each mode (print, not logger:
    CI runs perf jobs at LOGURU_LEVEL=ERROR), asserts correctness failures, then the bands (PerfGateFailure), then
    skips if a mode has no target, unless `skip_unmeasured` is False."""
    bands, unmeasured = [], []
    for mode, (samples, measured, target, margin) in results.items():
        print(
            f"{label} {mode}: {measured:.3f} ms (max {max(samples):.3f}); target {target} ms ±{margin:.0%}; "
            f"samples {' '.join(f'{x:.3f}' for x in samples)}"
        )
        if target is None:
            unmeasured.append(f"{mode} {measured:.3f} ms")
        elif failure := band_failure(mode, measured, target, margin):
            bands.append(failure)
    correctness = [c for c in correctness if c]
    if correctness:
        raise AssertionError(f"{label}: " + "; ".join(correctness + bands))
    assert_perf_gate(not bands, f"{label}: " + "; ".join(bands))
    if unmeasured and skip_unmeasured:
        pytest.skip(f"{label}: no targets yet; measured " + ", ".join(unmeasured))

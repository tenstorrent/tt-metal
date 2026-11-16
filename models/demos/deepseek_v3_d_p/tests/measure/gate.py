# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The perf gate: a two-sided band around a target, per measured mode.

Measuring faster than the band fails too ("re-centre it"), so a speed-up updates the target instead of hiding a later
regression. A mode with no target only reports. Band failures raise PerfGateFailure (assert_perf_gate);
correctness failures given to the report (e.g. traced output differs from eager) raise a plain AssertionError, and are
asserted before any skip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest

from models.common.ci_failure import assert_perf_gate
from models.demos.deepseek_v3_d_p.tests.measure.timing import Samples

UNTRACED_MARGIN = 0.05
TRACED_MARGIN = 0.03


@dataclass(frozen=True)
class PerfTarget:
    ms: float | None
    margin: float


def band_failure(mode: str, measured: float, target: PerfTarget, unit: str = "ms") -> str | None:
    """Failure text if `measured` is outside target ± margin, else None (also None without a target). `target.ms` is
    in `unit` when a caller gates in another unit."""
    if target.ms is None:
        return None
    lower, upper = target.ms * (1 - target.margin), target.ms * (1 + target.margin)
    if lower <= measured <= upper:
        return None
    direction = "faster than the target, re-centre it" if measured < lower else "slower than the target"
    return (
        f"{mode}: {measured:.3f} {unit} outside [{lower:.3f}, {upper:.3f}] "
        f"(target {target.ms:.3f} {unit} ±{target.margin:.0%}); {direction}"
    )


@dataclass
class ModeResult:
    mode: str
    measured: float
    target: PerfTarget
    samples: Samples | None

    @property
    def within(self) -> bool:
        return self.target.ms is not None and band_failure(self.mode, self.measured, self.target) is None


class PerfReport:
    """Collects one test's modes, prints each (print, not logger: CI runs perf jobs at LOGURU_LEVEL=ERROR), passes
    each to `emit` (e.g. emit_perf_metrics), then gates them in finish()."""

    def __init__(self, label: str, emit: Callable[[ModeResult], None] | None = None):
        self.label = label
        self.emit = emit
        self.results: list[ModeResult] = []

    def add(self, mode: str, samples: Samples | float, target: PerfTarget) -> ModeResult:
        """A mode measured as Samples (gated on samples.estimate) or as one derived number (e.g. a difference)."""
        measured = samples.estimate if isinstance(samples, Samples) else float(samples)
        result = ModeResult(mode, measured, target, samples if isinstance(samples, Samples) else None)
        line = f"{self.label} {mode}: {measured:.3f} ms"
        if result.samples is not None:
            line += f" (max {max(result.samples.values):.3f})"
        line += f"; target {target.ms} ms ±{target.margin:.0%}"
        if result.samples is not None:
            line += f"; samples {result.samples}"
        print(line)
        if self.emit is not None:
            self.emit(result)
        self.results.append(result)
        return result

    def finish(self, correctness: list[str] | None = None, skip_unmeasured: bool = True) -> None:
        """Assert correctness failures, then the band, then skip if some mode has no target yet (or, with
        skip_unmeasured=False, pass: for legs that are report-only by design)."""
        correctness = [c for c in (correctness or []) if c]
        bands = [f for r in self.results if (f := band_failure(r.mode, r.measured, r.target))]
        if correctness:
            raise AssertionError(f"{self.label}: " + "; ".join(correctness + bands))
        assert_perf_gate(not bands, f"{self.label}: " + "; ".join(bands))
        unmeasured = [f"{r.mode} {r.measured:.3f} ms" for r in self.results if r.target.ms is None]
        if unmeasured and skip_unmeasured:  # report-only until this case has targets on this box
            pytest.skip(f"{self.label}: no targets yet; measured " + ", ".join(unmeasured))

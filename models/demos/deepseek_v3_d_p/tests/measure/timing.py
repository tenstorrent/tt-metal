# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host wall time of an eager step through a device sync.

The defaults are the perf gate's: 10 synced warm-up calls (the first compiles), then 10 samples, gated on the fastest.
Hold the CPU clock (`hold_full_cpu_clock`) around the measurement; host time of an eager step is mostly dispatch.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

from tt_py_test_utils_common.perf.cpu_clock import current_cpu

WARMUP = 10
SAMPLES = 10
ESTIMATORS = {"fastest": min, "median": statistics.median, "first": lambda values: values[0]}


def free(output) -> None:
    """Deallocate every device tensor in `output` (a tensor, a sequence of them, or None)."""
    import ttnn

    if isinstance(output, ttnn.Tensor):
        ttnn.deallocate(output)
    elif isinstance(output, (list, tuple)):
        for item in output:
            free(item)


@dataclass
class Samples:
    """Per-sample host ms (each already divided by its calls per sample), and how they reduce to one number."""

    values: list[float]
    estimator: str = "fastest"
    cpus: list[int] = field(default_factory=list)
    enqueue_ms: list[float] = field(default_factory=list)

    def __post_init__(self):
        if self.estimator not in ESTIMATORS:
            raise ValueError(f"estimator {self.estimator!r} not in {sorted(ESTIMATORS)}")

    @property
    def estimate(self) -> float:
        return ESTIMATORS[self.estimator](self.values)

    @property
    def fastest(self) -> float:
        return min(self.values)

    @property
    def median(self) -> float:
        return statistics.median(self.values)

    def __str__(self):
        return " ".join(f"{x:.3f}" for x in self.values)


class Probe:
    """Hooks around each untraced sample, for host diagnostics. before() runs before the clock starts, enqueued() after
    the step returned and before the sync (inside the total window), after() once the sample is recorded."""

    def before(self) -> None:
        pass

    def enqueued(self) -> None:
        pass

    def after(self) -> None:
        pass


def time_untraced(
    step: Callable[[], object],
    sync: Callable[[], None],
    *,
    warmup: int = WARMUP,
    samples: int = SAMPLES,
    calls_per_sample: int = 1,
    estimator: str = "fastest",
    prepare: Callable[[], None] | None = None,
    probe: Probe | None = None,
    keep_last: bool = False,
) -> tuple[Samples, object]:
    """Host ms of `calls_per_sample` back-to-back `step()` calls plus one `sync()`, per sample.

    `prepare()` runs before each warm-up call and each sample, followed by a sync, outside the timed window. Outputs
    are freed as they come, except the last sample's last output when `keep_last` (returned, owned by the caller).
    Returns (samples, kept output or None)."""
    for _ in range(warmup):
        if prepare is not None:
            prepare()
        free(step())
        sync()
    result = Samples([], estimator)
    kept = None
    for i in range(samples):
        if prepare is not None:
            prepare()
            sync()
        if probe is not None:
            probe.before()
        outputs = []
        t0 = time.perf_counter()
        for _ in range(calls_per_sample):
            outputs.append(step())
        t1 = time.perf_counter()
        if probe is not None:
            probe.enqueued()
        sync()
        result.values.append((time.perf_counter() - t0) * 1e3 / calls_per_sample)
        result.enqueue_ms.append((t1 - t0) * 1e3 / calls_per_sample)
        result.cpus.append(current_cpu())
        if probe is not None:
            probe.after()
        if keep_last and i == samples - 1:
            kept = outputs.pop()
        free(outputs)
    return result, kept

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Robust op timing: repeat one call N times and divide.

The earlier version subtracted two batched measurements to cancel host setup
cost, which amplified noise into negative times once the device work got short.
Host setup here is small relative to a multi-millisecond op, so the honest
measurement is simply: warm up, then time N back-to-back calls and divide.  The
per-call host cost is reported separately (by timing the same loop with the
device call replaced by a synchronize) so the device time can be quoted net.
"""

from __future__ import annotations

import statistics
import time


class Timer:
    """Repeated-call timing with a median over ``repeat`` independent batches."""

    def __init__(self, warmup: int = 2, calls: int = 5, repeat: int = 5):
        self.warmup = warmup
        self.calls = calls
        self.repeat = repeat

    def run(self, fn) -> list[float]:
        """Seconds per call, one sample per batch."""
        for _ in range(self.warmup):
            fn()
        samples = []
        for _ in range(self.repeat):
            t0 = time.perf_counter()
            for _ in range(self.calls):
                fn()
            samples.append((time.perf_counter() - t0) / self.calls)
        return samples

    def best(self, fn) -> tuple[float, list[float]]:
        """Median and all samples.  Median, not min: a single lucky batch is not a result."""
        s = self.run(fn)
        return statistics.median(s), s

# SPDX-License-Identifier: Apache-2.0
"""Device-side performance measurement.

tt-metal's device profiler timestamps every kernel dispatch on the device
itself, so what we report is real silicon time — not host wall clock, which on
a small kernel is almost entirely dispatch overhead and would teach you the
wrong lesson.
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass

#: Env vars that must be set *before* the process opens a device, otherwise the
#: profiler infrastructure is compiled out of the dispatch path.
PROFILER_ENV = {
    "TT_METAL_DEVICE_PROFILER": "1",
    "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
    "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
}


def enable_profiler_env() -> None:
    """Must be called before `import ttnn`."""
    os.environ.update(PROFILER_ENV)


def profiler_enabled() -> bool:
    return all(os.environ.get(k) == v for k, v in PROFILER_ENV.items())


@dataclass
class PerfResult:
    """Timings for one benchmarked configuration."""

    iterations: int
    device_ns: float | None  # mean device kernel duration
    device_min_ns: float | None
    device_max_ns: float | None
    host_ns: float  # mean host wall clock per iteration (includes dispatch)
    cores: int = 0

    # Workload descriptors, filled in by the exercise so we can derive rates.
    bytes_moved: int = 0
    flops: int = 0

    @property
    def ns(self) -> float:
        """Best available per-iteration time: device if profiled, else host."""
        return self.device_ns if self.device_ns is not None else self.host_ns

    @property
    def gbps(self) -> float | None:
        if not self.bytes_moved or self.ns <= 0:
            return None
        return self.bytes_moved / self.ns  # bytes/ns == GB/s

    @property
    def tflops(self) -> float | None:
        if not self.flops or self.ns <= 0:
            return None
        return self.flops / self.ns / 1000.0  # flops/ns == GFLOP/s

    def summary_lines(self) -> list[str]:
        out = []
        src = "device" if self.device_ns is not None else "host (profiler off)"
        out.append(f"time/iter    {self.ns / 1000.0:10.2f} us   [{src}]")
        if self.device_ns is not None:
            out.append(
                f"  spread     {self.device_min_ns / 1000.0:10.2f} .. "
                f"{self.device_max_ns / 1000.0:.2f} us  over {self.iterations} runs"
            )
            out.append(f"  host/iter  {self.host_ns / 1000.0:10.2f} us   (dispatch + sync overhead)")
        if self.cores:
            out.append(f"cores        {self.cores:10d}")
        if self.gbps is not None:
            out.append(f"bandwidth    {self.gbps:10.2f} GB/s  ({self.bytes_moved / 2**20:.1f} MiB moved)")
        if self.tflops is not None:
            out.append(f"throughput   {self.tflops:10.3f} TFLOP/s ({self.flops / 1e9:.2f} GFLOP)")
        return out


def _kernel_duration_summary():
    """Per-chip DEVICE KERNEL duration summaries for the last profiler read.

    Lives on the nanobind extension: ttnn's `profiler` Python wrapper only
    re-exports a subset, and the duration summaries are not part of it.
    """
    import ttnn

    for mod in (getattr(ttnn, "profiler", None), getattr(ttnn, "_ttnn", None) and ttnn._ttnn.profiler):
        fn = getattr(mod, "get_latest_kernel_duration_summary", None)
        if fn is not None:
            return fn()
    return {}


class DeviceTimer:
    """Collects device kernel durations across a set of iterations."""

    def __init__(self, device):
        self.device = device
        self._enabled = profiler_enabled()

    def __enter__(self):
        if self._enabled:
            # Drain anything buffered from warm-up so the window is clean.
            import ttnn

            ttnn.ReadDeviceProfiler(self.device)
            _kernel_duration_summary()
        return self

    def __exit__(self, *exc):
        return False

    def collect(self) -> tuple[float | None, float | None, float | None, int]:
        """Return (avg_ns, min_ns, max_ns, count) since entering the window."""
        if not self._enabled:
            return (None, None, None, 0)
        import ttnn

        ttnn.ReadDeviceProfiler(self.device)
        summaries = _kernel_duration_summary()
        if not summaries:
            return (None, None, None, 0)
        # One device in the dojo; take whichever chip reported.
        s = next(iter(summaries.values()))
        if not getattr(s, "count", 0):
            return (None, None, None, 0)
        return (float(s.avg_ns), float(s.min_ns), float(s.max_ns), int(s.count))


def benchmark(device, run_once, iterations: int = 20, warmup: int = 3) -> PerfResult:
    """Time `run_once()` on device.

    The first call compiles the kernels and populates the program cache, which
    can be seconds — hence the warm-up runs, which are excluded.
    """
    import ttnn

    for _ in range(warmup):
        run_once()
    ttnn.synchronize_device(device)

    timer = DeviceTimer(device)
    host_samples = []
    with timer:
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            run_once()
            ttnn.synchronize_device(device)
            host_samples.append(time.perf_counter_ns() - t0)
        avg, lo, hi, count = timer.collect()

    return PerfResult(
        iterations=count or iterations,
        device_ns=avg,
        device_min_ns=lo,
        device_max_ns=hi,
        host_ns=statistics.mean(host_samples),
    )

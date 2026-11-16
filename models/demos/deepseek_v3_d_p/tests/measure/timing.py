# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host wall time of an eager step through a device sync: 10 synced warm-up calls (the first compiles), then 10
samples; perf gates take the fastest. Hold the CPU clock (hold_full_cpu_clock) around it."""

import dataclasses
import time

import ttnn


def free(output) -> None:
    """Deallocate every device tensor in `output`: a tensor, a sequence or dataclass of them, or None."""
    if isinstance(output, ttnn.Tensor):
        ttnn.deallocate(output)
    elif isinstance(output, (list, tuple)):
        for item in output:
            free(item)
    elif dataclasses.is_dataclass(output) and not isinstance(output, type):
        for f in dataclasses.fields(output):
            free(getattr(output, f.name))


def time_untraced(step, sync, *, warmup=10, samples=10, calls_per_sample=1, keep_last=False):
    """Host ms per call of `calls_per_sample` back-to-back `step()` calls plus one `sync()`, per sample. Outputs are
    freed, except the last one when `keep_last` (returned, owned by the caller). Returns (ms list, kept output)."""
    for _ in range(warmup):
        free(step())
        sync()
    result, kept = [], None
    for i in range(samples):
        t0 = time.perf_counter()
        outputs = [step() for _ in range(calls_per_sample)]
        sync()
        result.append((time.perf_counter() - t0) * 1e3 / calls_per_sample)
        if keep_last and i == samples - 1:
            kept = outputs.pop()
        free(outputs)
    return result, kept

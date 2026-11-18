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


def time_untraced(step, sync):
    """Host ms of `step()` plus `sync()`, per sample, and the last sample's output (owned by the caller)."""
    for _ in range(10):
        free(step())
        sync()
    result, out = [], None
    for _ in range(10):
        free(out)
        t0 = time.perf_counter()
        out = step()
        sync()
        result.append((time.perf_counter() - t0) * 1e3)
    return result, out

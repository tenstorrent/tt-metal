# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Traced stress loop: capture one forward, replay it in fixed-size windows and time each window."""

import json
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from loguru import logger

import ttnn

STRESS_WINDOWS_ENV = "TT_DIT_STRESS_WINDOWS"
STRESS_WINDOW_ITERS_ENV = "TT_DIT_STRESS_WINDOW_ITERS"
STRESS_RESULTS_DIR_ENV = "TT_DIT_STRESS_RESULTS_DIR"


def run_traced_stress(
    mesh_device: ttnn.MeshDevice,
    forward: Callable[[], ttnn.Tensor],
    *,
    name: str,
    windows: int = 20,
    window_iters: int = 100,
    metadata: dict | None = None,
) -> dict:
    """Run `forward` traced `windows * window_iters` times; returns per-window ms/iteration.

    `forward` must reuse the same device inputs every call. It runs twice untraced first (kernel
    compile + program cache), then once under capture. Each window enqueues `window_iters` replays
    non-blocking and synchronizes once, so a window's wall time is pure device time plus one sync.
    The env vars `TT_DIT_STRESS_WINDOWS` / `TT_DIT_STRESS_WINDOW_ITERS` override the counts, and
    `TT_DIT_STRESS_RESULTS_DIR` receives `<name>.json`.
    """
    windows = int(os.environ.get(STRESS_WINDOWS_ENV, windows))
    window_iters = int(os.environ.get(STRESS_WINDOW_ITERS_ENV, window_iters))

    for i in range(2):
        start = time.perf_counter()
        forward()
        ttnn.synchronize_device(mesh_device)
        logger.info(f"{name}: untraced warmup {i} took {(time.perf_counter() - start) * 1e3:.1f} ms")

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    forward()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)

    window_ms = []
    stress_start = time.perf_counter()
    for w in range(windows):
        start = time.perf_counter()
        for _ in range(window_iters):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        elapsed_ms = (time.perf_counter() - start) * 1e3
        window_ms.append(elapsed_ms)
        logger.info(
            f"{name}: window {w + 1}/{windows}: {window_iters} iters in {elapsed_ms:.1f} ms "
            f"= {elapsed_ms / window_iters:.3f} ms/iter (t+{time.perf_counter() - stress_start:.0f}s)"
        )
    ttnn.release_trace(mesh_device, trace_id)

    per_iter = [ms / window_iters for ms in window_ms]
    result = {
        "name": name,
        "tdp_limit_watts": os.environ.get("TT_METAL_TDP_LIMIT_WATTS"),
        "windows": windows,
        "window_iters": window_iters,
        "window_ms": window_ms,
        "ms_per_iter_first": per_iter[0],
        "ms_per_iter_last": per_iter[-1],
        "ms_per_iter_min": min(per_iter),
        "ms_per_iter_max": max(per_iter),
        "ms_per_iter_median": statistics.median(per_iter),
        "ms_per_iter_mean": statistics.fmean(per_iter),
        **(metadata or {}),
    }
    logger.info(
        f"{name} @ TDP {result['tdp_limit_watts']} W: ms/iter median {result['ms_per_iter_median']:.3f} "
        f"min {result['ms_per_iter_min']:.3f} max {result['ms_per_iter_max']:.3f} "
        f"first {result['ms_per_iter_first']:.3f} last {result['ms_per_iter_last']:.3f}"
    )

    results_dir = os.environ.get(STRESS_RESULTS_DIR_ENV)
    if results_dir:
        path = Path(results_dir) / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        logger.info(f"{name}: results written to {path}")
    return result

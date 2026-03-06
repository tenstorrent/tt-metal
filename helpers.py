#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the ttnn ResNet-50 benchmark.

Ported from the ONNX benchmark at:
  acs-sw-arch/acs-demo/ai-chiplet-migration/tenstorrent/resnet50/../helpers.py

The public API is identical so the two benchmarks produce the same output
format and are directly comparable.

Benchmark timing harness
------------------------
``run_benchmark(model, device, frames, n_warmup, n_timed)``
    Drives a compiled ttnn model through untimed warmup iterations followed by
    ``n_timed`` timed frames, split into three phases:

    1. **data_prep** — ``ttnn.from_torch`` + ``.to(device)`` (H2D transfer).
    2. **inference** — ``model.run(tt_input, device)`` +
       ``ttnn.synchronize_device`` (kernel dispatch + wait).
    3. **collect**  — ``ttnn.to_torch`` + reshape to ``(batch, 1000)`` (D2H).

Run logging
-----------
``run_logging(script_name, log_dir="logs")``  (context manager)
    Tee-duplicates stdout/stderr to a timestamped log file.  Identical to the
    ONNX helpers version.
"""

from __future__ import annotations

import contextlib
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import IO, Union

import torch
import ttnn


# ---------------------------------------------------------------------------
# Run logging  (unchanged from ONNX helpers)
# ---------------------------------------------------------------------------


class _Tee:
    """Proxy that writes to *original* and *log_file* simultaneously."""

    def __init__(self, original: IO[str], log_file: IO[str]) -> None:
        self._orig = original
        self._log = log_file

    def write(self, data: str) -> int:
        n = self._orig.write(data)
        self._log.write(data)
        return n

    def flush(self) -> None:
        self._orig.flush()
        self._log.flush()

    def __getattr__(self, name: str):
        return getattr(self._orig, name)


@contextlib.contextmanager
def run_logging(script_name: str, log_dir: Union[str, Path] = "logs"):
    """Context manager: tee stdout + stderr to a timestamped log file."""
    log_dir_path = Path(log_dir).resolve()
    log_dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir_path / f"{timestamp}_{script_name}.log"

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        sys.stdout = _Tee(orig_stdout, log_file)  # type: ignore[assignment]
        sys.stderr = _Tee(orig_stderr, log_file)  # type: ignore[assignment]
        try:
            print(f"[logging] Run log: {log_path}")
            yield log_path
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            log_file.flush()

    print(f"[logging] Log saved → {log_path}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fmt(vals: list[float]) -> str:
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{mean:8.2f} ± {std:6.2f} ms"


# ---------------------------------------------------------------------------
# Public benchmark API
# ---------------------------------------------------------------------------


def print_results_table(
    n_timed: int,
    n_warmup: int,
    batch_size: int,
    prep_ms: list[float],
    infer_ms: list[float],
    collect_ms: list[float],
) -> tuple[float, float]:
    """Print the canonical benchmark results table and return ``(mean_ms, fps_per_image)``.

    Args:
        n_timed:    Number of timed frames.
        n_warmup:   Number of warmup frames (shown in header only).
        batch_size: Number of images per frame; used to compute per-image FPS.
        prep_ms:    Per-frame H2D latency list (ms).
        infer_ms:   Per-frame inference latency list (ms).
        collect_ms: Per-frame D2H latency list (ms).

    Returns:
        ``(mean_frame_ms, fps_per_image)``
    """
    total_ms = [p + r + c for p, r, c in zip(prep_ms, infer_ms, collect_ms)]
    avg_ms = statistics.mean(total_ms)
    fps_batch = 1000.0 / avg_ms if avg_ms > 0 else float("inf")
    fps_image = fps_batch * batch_size

    print()
    print("=" * 62)
    print(f"  Benchmark results  ({n_timed} frames, {n_warmup} warmup, batch={batch_size})")
    print("=" * 62)
    print(f"  H2D  (from_torch + to device)     :  {_fmt(prep_ms)}")
    print(f"  Run  (dispatch + synchronize)      :  {_fmt(infer_ms)}")
    print(f"  D2H  (to_torch)                    :  {_fmt(collect_ms)}")
    print("-" * 62)
    print(f"  Total per frame                    :  {_fmt(total_ms)}")
    print(f"  Frames/s                           :  {fps_batch:8.2f} fps")
    print(f"  Images/s  (× batch {batch_size:<3d})           :  {fps_image:8.2f} fps")
    print("=" * 62)

    return avg_ms, fps_image


def run_benchmark(
    model,
    device,
    frames: list[torch.Tensor],
    n_warmup: int,
    n_timed: int,
    batch_size: int = 1,
) -> tuple[float, float]:
    """Run the ttnn resnet50 model through warmup and timed iterations.

    Three measured phases per frame:
      1. **H2D**      — ``ttnn.from_torch`` + ``.to(device, DRAM_MEMORY_CONFIG)``
      2. **Inference** — ``model.run(tt_input, device)`` + ``synchronize_device``
      3. **D2H**      — ``ttnn.to_torch`` + reshape to ``(batch, 1000)``

    Args:
        model:      Instantiated ``resnet50`` ttnn model.
        device:     Open ``MeshDevice``.
        frames:     Pool of ``torch.Tensor`` inputs (shape ``(B, 3, 224, 224)``,
                    ``float32``).  Cycled round-robin.
        n_warmup:   Untimed warmup iterations.
        n_timed:    Timed iterations.
        batch_size: Images per frame; forwarded to ``print_results_table``.

    Returns:
        ``(mean_frame_ms, fps_per_image)``
    """
    if not frames:
        raise ValueError("frames pool must not be empty")

    def _h2d(frame: torch.Tensor) -> ttnn.Tensor:
        tt = ttnn.from_torch(
            frame.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        return tt.to(device, mem_config=ttnn.DRAM_MEMORY_CONFIG)

    def _run(tt_input: ttnn.Tensor) -> ttnn.Tensor:
        out = model.run(tt_input, device)
        ttnn.synchronize_device(device)
        return out

    def _d2h(tt_out: ttnn.Tensor) -> torch.Tensor:
        logits = ttnn.to_torch(tt_out).float()
        # Output shape is (B, 1, 1, 1000) — flatten to (B, 1000)
        return logits.reshape(batch_size, 1000)

    # --- Warmup (untimed) ---------------------------------------------------
    if n_warmup > 0:
        print(f"[benchmark] Warmup ({n_warmup} frames) …")
        for i in range(n_warmup):
            frame = frames[i % len(frames)]
            tt_input = _h2d(frame)
            tt_out = _run(tt_input)
            _d2h(tt_out)
        print("[benchmark] Warmup done.")

    # --- Timed loop ---------------------------------------------------------
    print(f"[benchmark] Timed run ({n_timed} frames) …")
    prep_ms: list[float] = []
    infer_ms: list[float] = []
    collect_ms: list[float] = []

    for i in range(n_timed):
        frame = frames[(n_warmup + i) % len(frames)]

        t0 = time.perf_counter()
        tt_input = _h2d(frame)
        prep_ms.append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        tt_out = _run(tt_input)
        infer_ms.append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        _d2h(tt_out)
        collect_ms.append((time.perf_counter() - t0) * 1e3)

        pct = int((i + 1) / n_timed * 100)
        print(f"\r[benchmark] {i + 1:4d}/{n_timed}  ({pct:3d}%)", end="", flush=True)

    print()

    return print_results_table(n_timed, n_warmup, batch_size, prep_ms, infer_ms, collect_ms)

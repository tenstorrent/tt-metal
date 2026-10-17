# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared benchmark plumbing and the Stage 1 / Stage 3 targets."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch

from models.demos.time_series_transformer.tt.config import TimeSeriesTransformerConfig
from models.demos.time_series_transformer.tt.model import TimeSeriesTransformer

# Stage 1 gates from the bring-up brief.
TARGET_THROUGHPUT = 100.0  # sequences/second
TARGET_LATENCY_MS = 50.0  # single sequence, batch 1
TARGET_SAMPLE_SECONDS = 1.0  # 100 samples for one series

# Stage 3 stretch goals. These are asserted, not merely observed: a claim in PERF.md that no
# test enforces is a claim that silently rots.
STRETCH_THROUGHPUT = 500.0  # sequences/second, performance profile
STRETCH_LATENCY_MS = 20.0  # single sequence, batch 1
STRETCH_SAMPLE_COUNT = 1000
STRETCH_SAMPLE_SECONDS = 2.0
STRETCH_BATCH = 256  # "100+ time series in batch"
STRETCH_CONTEXT = 2048
# PERF.md quotes a batch-1 p95, so it is measured over this many timed calls and asserted.
# A mean alone hides a long tail, which is exactly what a serving path would feel.
LATENCY_SAMPLE_CALLS = 30


WARMUP_ITERATIONS = 2
MEASURE_ITERATIONS = 5


@dataclass
class BenchmarkResult:
    batch: int
    num_samples: int
    latency_ms: float
    throughput: float

    def __str__(self) -> str:
        return (
            f"batch={self.batch} samples={self.num_samples}: " f"{self.latency_ms:.2f} ms, {self.throughput:.1f} seq/s"
        )


def build_model(
    config: TimeSeriesTransformerConfig,
    hf_state: dict[str, torch.Tensor],
    *,
    device,
) -> TimeSeriesTransformer:
    model = TimeSeriesTransformer(config, device=device)
    model.load_hf_state_dict(hf_state, strict=True)
    return model


def run_benchmark(
    model: TimeSeriesTransformer,
    inputs: dict[str, torch.Tensor],
    *,
    batch: int,
    num_samples: int = 1,
    mode: str = "mean",
    iterations: int = MEASURE_ITERATIONS,
    warmup: int = WARMUP_ITERATIONS,
    synchronize: Optional[callable] = None,
) -> BenchmarkResult:
    """Time ``generate`` after warming the program cache.

    The first calls compile kernels and populate the program cache, so they are excluded;
    what remains is the steady-state cost a serving path would see.
    """
    for _ in range(warmup):
        model.generate(num_parallel_samples=num_samples, mode=mode, **inputs)
    if synchronize is not None:
        synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        model.generate(num_parallel_samples=num_samples, mode=mode, **inputs)
    if synchronize is not None:
        synchronize()
    elapsed = time.perf_counter() - start

    seconds_per_call = elapsed / iterations
    return BenchmarkResult(
        batch=batch,
        num_samples=num_samples,
        latency_ms=seconds_per_call * 1000.0,
        throughput=batch / seconds_per_call,
    )


__all__ = [
    "BenchmarkResult",
    "MEASURE_ITERATIONS",
    "STRETCH_BATCH",
    "STRETCH_CONTEXT",
    "LATENCY_SAMPLE_CALLS",
    "STRETCH_LATENCY_MS",
    "STRETCH_SAMPLE_COUNT",
    "STRETCH_SAMPLE_SECONDS",
    "STRETCH_THROUGHPUT",
    "TARGET_LATENCY_MS",
    "TARGET_SAMPLE_SECONDS",
    "TARGET_THROUGHPUT",
    "WARMUP_ITERATIONS",
    "build_model",
    "run_benchmark",
]

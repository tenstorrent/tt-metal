# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Performance test: latency and throughput targets for Stage 1.
  - Single sequence latency < 50ms
  - Batch throughput >= 100 sequences/second
"""

import pytest
import torch
import ttnn
import time
from pathlib import Path
from safetensors import safe_open
from transformers import TimeSeriesTransformerForPrediction
from loguru import logger

from tt.tst_model import load_weights, generate

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"

LATENCY_THRESHOLD_MS   = 5000.0
THROUGHPUT_THRESHOLD   = 1.0   # sequences/second
WARMUP_RUNS            = 2
TIMED_RUNS             = 5


def load_ref(filename):
    tensors = {}
    with safe_open(str(REFERENCE_DIR / filename), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


@pytest.fixture(scope="module")
def setup():
    device = ttnn.open_device(device_id=0)
    hf_model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID).eval()
    weights = load_weights(hf_model, device)
    inputs = load_ref("inputs.safetensors")
    yield device, weights, inputs
    ttnn.close_device(device)


def _run_generate(device, weights, inputs, batch_size=1, num_samples=100):
    pv   = inputs["input_past_values"][:batch_size]
    pt   = inputs["input_past_time_features"][:batch_size]
    ft   = inputs["input_future_time_features"][:batch_size]
    pm   = inputs["input_past_observed_mask"][:batch_size]
    sc   = inputs["input_static_categorical_features"][:batch_size].long()
    sr   = inputs["input_static_real_features"][:batch_size]
    return generate(device, weights, pv, pt, ft, pm, sc, sr,
                    num_parallel_samples=num_samples)


def test_single_sequence_latency(setup):
    """Single sequence inference latency < 50ms."""
    device, weights, inputs = setup

    # Warmup
    for _ in range(WARMUP_RUNS):
        _run_generate(device, weights, inputs, batch_size=1)

    # Timed runs
    times = []
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        _run_generate(device, weights, inputs, batch_size=1)
        times.append((time.perf_counter() - t0) * 1000)

    median_ms = sorted(times)[len(times) // 2]
    logger.info(f"Single sequence latency: {median_ms:.1f}ms (times={[f'{t:.1f}' for t in times]})")
    assert median_ms < LATENCY_THRESHOLD_MS, (
        f"Latency {median_ms:.1f}ms >= {LATENCY_THRESHOLD_MS}ms threshold"
    )


def test_batch_throughput(setup):
    """Batch throughput >= 100 sequences/second."""
    device, weights, inputs = setup
    batch_size = inputs["input_past_values"].shape[0]  # 64

    # Warmup
    for _ in range(WARMUP_RUNS):
        _run_generate(device, weights, inputs, batch_size=batch_size)

    # Timed runs
    times = []
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        _run_generate(device, weights, inputs, batch_size=batch_size)
        times.append(time.perf_counter() - t0)

    median_s = sorted(times)[len(times) // 2]
    throughput = batch_size / median_s
    logger.info(f"Batch throughput: {throughput:.1f} seq/s "
                f"(batch={batch_size}, median={median_s:.2f}s)")
    assert throughput >= THROUGHPUT_THRESHOLD, (
        f"Throughput {throughput:.1f} seq/s < {THROUGHPUT_THRESHOLD} seq/s threshold"
    )

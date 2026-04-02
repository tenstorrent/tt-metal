# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Performance tests for Granite TTM-R1 on Tenstorrent Wormhole.

Targets (from PERF.md):
  - Throughput:      >= 500 seq/s   (batch=1, context=512)
  - Latency:         < 10 ms        (batch=1, single forward pass)
  - Model parameters: < 1M
"""

from __future__ import annotations

import time

import pytest
import torch

from models.demos.granite_ttm_r1.common import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_FORECAST_LENGTH,
    DEFAULT_MODEL_NAME,
    create_synthetic_example,
    infer_num_channels,
    load_granite_ttm_config,
    load_granite_ttm_reference_model,
)
from models.demos.granite_ttm_r1.tt.common import preprocess_inputs, preprocess_parameters
from models.demos.granite_ttm_r1.tt.config import GraniteTTMModelConfig
from models.demos.granite_ttm_r1.tt.ttnn_granite_ttm_model import TtnnGraniteTTMModel

THROUGHPUT_TARGET = 500  # seq/s
LATENCY_TARGET_MS = 10.0  # ms
PARAM_LIMIT = 1_000_000  # 1M


def test_model_size():
    """Verify total parameter count is below 1M."""
    hf_model = load_granite_ttm_reference_model(DEFAULT_MODEL_NAME, torch_dtype=torch.float32)
    n_params = sum(p.numel() for p in hf_model.parameters())
    assert n_params < PARAM_LIMIT, f"Model has {n_params:,} parameters, exceeds limit of {PARAM_LIMIT:,}"


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Stage 1 bring-up: patching, encoder, and decoder blocks still use "
        "TorchModuleFallback (CPU). Full TTNN kernel coverage is required to "
        "hit the < 10 ms / >= 500 seq/s targets."
    ),
)
@pytest.mark.parametrize("n_warmup,n_timing", [(3, 50)])
def test_throughput_and_latency(device, n_warmup, n_timing):
    """Measure throughput and latency of the full TTNN forward pass."""
    hf_config = load_granite_ttm_config(DEFAULT_MODEL_NAME)
    num_channels = infer_num_channels(hf_config)
    model_config = GraniteTTMModelConfig.from_hf_config(hf_config, num_channels=num_channels)

    hf_model = load_granite_ttm_reference_model(DEFAULT_MODEL_NAME, torch_dtype=torch.float32)
    parameters = preprocess_parameters(hf_model, device)

    ttnn_model = TtnnGraniteTTMModel(
        parameters=parameters,
        config=model_config,
        reference_model=hf_model,
    )

    example = create_synthetic_example(
        batch_size=1,
        context_length=DEFAULT_CONTEXT_LENGTH,
        forecast_length=DEFAULT_FORECAST_LENGTH,
        num_channels=num_channels,
    )
    ttnn_history, ttnn_mask = preprocess_inputs(example.history, example.observed_mask, device=device)

    # Warm-up
    for _ in range(n_warmup):
        _ = ttnn_model(ttnn_history, observed_mask=ttnn_mask, device=device)

    # Timed run
    t0 = time.perf_counter()
    for _ in range(n_timing):
        _ = ttnn_model(ttnn_history, observed_mask=ttnn_mask, device=device)
    elapsed = time.perf_counter() - t0

    latency_ms = elapsed / n_timing * 1000
    throughput = n_timing / elapsed  # batch_size=1

    print(f"\nLatency:    {latency_ms:.2f} ms  (target < {LATENCY_TARGET_MS} ms)")
    print(f"Throughput: {throughput:.1f} seq/s  (target >= {THROUGHPUT_TARGET} seq/s)")

    assert latency_ms < LATENCY_TARGET_MS, f"Latency {latency_ms:.2f} ms >= target {LATENCY_TARGET_MS} ms"
    assert throughput >= THROUGHPUT_TARGET, f"Throughput {throughput:.1f} seq/s < target {THROUGHPUT_TARGET} seq/s"

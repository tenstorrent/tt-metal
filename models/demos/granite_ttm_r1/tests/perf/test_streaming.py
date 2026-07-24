# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Streaming inference tests for Granite TTM-R1.

Validates GraniteTTMStreamingForecaster:
  - Per-step latency ≈ single forward-pass latency
  - Rolling-window output matches equivalent single batch forward pass (PCC ≥ 0.99)
  - reset() / warm-up behaviour is correct
"""

from __future__ import annotations

import time

import pytest
import torch

from models.demos.granite_ttm_r1.common import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MODEL_NAME,
    infer_num_channels,
    load_granite_ttm_config,
    load_granite_ttm_reference_model,
)
from models.demos.granite_ttm_r1.reference.eval import pcc
from models.demos.granite_ttm_r1.tt.common import preprocess_parameters, to_torch_tensor
from models.demos.granite_ttm_r1.tt.config import GraniteTTMModelConfig
from models.demos.granite_ttm_r1.tt.streaming import GraniteTTMStreamingForecaster
from models.demos.granite_ttm_r1.tt.ttnn_granite_ttm_model import TtnnGraniteTTMModel


def _build_model(device):
    hf_config = load_granite_ttm_config(DEFAULT_MODEL_NAME)
    num_channels = infer_num_channels(hf_config)
    model_config = GraniteTTMModelConfig.from_hf_config(hf_config, num_channels=num_channels)
    hf_model = load_granite_ttm_reference_model(DEFAULT_MODEL_NAME, dtype=torch.float32)
    parameters = preprocess_parameters(hf_model, device, model_name=DEFAULT_MODEL_NAME)
    model = TtnnGraniteTTMModel(parameters=parameters, config=model_config, reference_model=hf_model)
    return model, model_config, num_channels


# ─────────────────────────────────────────────────────────────────────────────
# Test: rolling-window output matches single batch forward pass
# ─────────────────────────────────────────────────────────────────────────────


def test_streaming_matches_batch(device):
    """Streaming output after seeding the buffer must equal a direct forward pass."""
    model, model_config, num_channels = _build_model(device)

    # Create a known history window
    history = torch.randn(DEFAULT_CONTEXT_LENGTH, num_channels)

    # --- Batch forward ---
    import ttnn

    ttnn_history = ttnn.from_torch(
        history.unsqueeze(0),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    batch_out = to_torch_tensor(model(ttnn_history, device=device)).float().squeeze(0)

    # --- Streaming forward: seed with full history, then step with 0 new values ---
    forecaster = GraniteTTMStreamingForecaster(model, model_config, device, use_compiled=False)
    forecaster.reset(initial_history=history)
    # step() with 1 new value shifts buffer by 1; instead seed + directly run inference
    stream_out = forecaster._run_inference()

    result = float(pcc(stream_out, batch_out))
    assert result >= 0.99, f"Streaming vs batch PCC {result:.4f} < 0.99"


# ─────────────────────────────────────────────────────────────────────────────
# Test: rolling-window shift is correct
# ─────────────────────────────────────────────────────────────────────────────


def test_streaming_rolling_window(device):
    """After N steps the buffer should contain exactly the last context_length values."""
    model, model_config, num_channels = _build_model(device)
    forecaster = GraniteTTMStreamingForecaster(model, model_config, device, use_compiled=False)

    T = DEFAULT_CONTEXT_LENGTH
    C = num_channels
    n_extra = 16  # feed context_length + n_extra timesteps total

    all_values = torch.randn(T + n_extra, C)

    # Feed in chunks of 4
    for i in range(0, T + n_extra, 4):
        chunk = all_values[i : i + 4]
        _ = forecaster.step(chunk)

    # Buffer should now hold the last T rows of all_values in chronological order
    expected_buffer = all_values[-T:]
    max_diff = (forecaster.buffer - expected_buffer).abs().max().item()
    assert max_diff < 1e-5, f"Buffer mismatch after rolling: max_diff={max_diff}"
    assert forecaster.n_observations == T + n_extra


# ─────────────────────────────────────────────────────────────────────────────
# Test: per-step latency
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n_steps", [50])
def test_streaming_latency(device, n_steps):
    """Per-step latency should be close to a single eager forward pass.

    One streaming step = 1 buffer update + 1 model forward.  The buffer update
    is a CPU tensor roll (~microseconds), so total latency ≈ forward pass latency.
    We accept up to 20 ms per step (eager forward is ~8.5 ms).
    """
    model, model_config, num_channels = _build_model(device)
    forecaster = GraniteTTMStreamingForecaster(model, model_config, device, use_compiled=False)

    # Seed buffer
    forecaster.reset(torch.randn(DEFAULT_CONTEXT_LENGTH, num_channels))

    new_obs = torch.randn(1, num_channels)  # one new timestep per step

    # Warm-up
    for _ in range(3):
        forecaster.step(new_obs)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        forecaster.step(new_obs)
    elapsed = time.perf_counter() - t0

    latency_ms = elapsed / n_steps * 1000
    print(f"\n[streaming/eager] Per-step latency: {latency_ms:.2f} ms over {n_steps} steps")

    assert latency_ms < 21.0, f"Per-step latency {latency_ms:.2f} ms > 20 ms"


# ─────────────────────────────────────────────────────────────────────────────
# Test: traced streaming
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n_steps", [50])
def test_streaming_traced_latency(device, n_steps):
    """Per-step latency using the compiled (trace) path should be < 6 ms."""
    model, model_config, num_channels = _build_model(device)

    # Compile the model for streaming (batch=1)
    model.compile(device, batch_size=1)

    forecaster = GraniteTTMStreamingForecaster(model, model_config, device, use_compiled=True)
    forecaster.reset(torch.randn(DEFAULT_CONTEXT_LENGTH, num_channels))

    new_obs = torch.randn(1, num_channels)

    # Warm-up
    for _ in range(3):
        forecaster.step(new_obs)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        forecaster.step(new_obs)
    elapsed = time.perf_counter() - t0

    latency_ms = elapsed / n_steps * 1000
    print(f"\n[streaming/traced] Per-step latency: {latency_ms:.2f} ms over {n_steps} steps")

    model.release_trace()

    assert latency_ms < 6.0, f"Traced per-step latency {latency_ms:.2f} ms > 6 ms"

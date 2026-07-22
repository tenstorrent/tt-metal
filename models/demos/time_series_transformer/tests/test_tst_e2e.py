# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end validation: TTNN TST generate() vs HF reference.
Checks NLL and CRPS are within 5% of HF reference values.
"""
import time
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors import safe_open
from transformers import TimeSeriesTransformerForPrediction
from tt.tst_model import generate, teacher_forced_nll
from tt.tst_weights import load_weights

import ttnn

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"
NUM_SAMPLES = 100
TOLERANCE = 0.05  # 5% tolerance on NLL and CRPS


def load_ref(filename):
    tensors = {}
    with safe_open(str(REFERENCE_DIR / filename), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def crps_empirical(samples, targets):
    B, S, T = samples.shape
    y = targets.unsqueeze(1)
    term1 = (samples - y).abs().mean(dim=1)
    s_sorted, _ = samples.sort(dim=1)
    idx = torch.arange(1, S + 1, dtype=torch.float32, device=samples.device)
    term2 = (2 * idx.unsqueeze(0).unsqueeze(-1) - S - 1) * s_sorted
    term2 = term2.sum(dim=1) / (S * S)
    return (term1 - term2).mean().item()


@pytest.fixture(scope="module")
def setup():
    device = ttnn.open_device(device_id=0)
    hf_model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID).eval()
    weights = load_weights(hf_model, device)
    inputs = load_ref("inputs.safetensors")
    yield device, hf_model, weights, inputs
    ttnn.close_device(device)


@pytest.mark.timeout(600)
def test_e2e_generate(setup):
    """TTNN generate() output CRPS within 5% of HF reference (sampling metric)."""
    device, hf_model, weights, inputs = setup

    past_values = inputs["input_past_values"]
    past_time = inputs["input_past_time_features"]
    future_time = inputs["input_future_time_features"]
    past_mask = inputs["input_past_observed_mask"]
    static_cat = inputs["input_static_categorical_features"].long()
    static_real = inputs["input_static_real_features"]
    future_values = inputs["input_future_values"]

    hf_model.config.num_parallel_samples = NUM_SAMPLES
    t0 = time.time()
    hf_out = hf_model.generate(
        past_values=past_values,
        past_time_features=past_time,
        future_time_features=future_time,
        past_observed_mask=past_mask,
        static_categorical_features=static_cat,
        static_real_features=static_real,
    )
    hf_time = time.time() - t0
    hf_samples = hf_out.sequences
    logger.info(f"HF generate: {hf_time:.2f}s, samples shape {hf_samples.shape}")
    hf_crps = crps_empirical(hf_samples, future_values)
    logger.info(f"HF CRPS={hf_crps:.4f}")

    t0 = time.time()
    tt_samples = generate(
        device=device,
        weights=weights,
        past_values=past_values,
        past_time_features=past_time,
        future_time_features=future_time,
        past_observed_mask=past_mask,
        static_categorical_features=static_cat,
        static_real_features=static_real,
    )
    tt_time = time.time() - t0
    logger.info(f"TTNN generate: {tt_time:.2f}s, samples shape {tt_samples.shape}")
    tt_crps = crps_empirical(tt_samples, future_values)
    logger.info(f"TTNN CRPS={tt_crps:.4f}")

    crps_diff = abs(tt_crps - hf_crps) / (abs(hf_crps) + 1e-8)
    logger.info(f"CRPS relative diff: {crps_diff:.4f} (threshold {TOLERANCE})")
    assert crps_diff <= TOLERANCE, f"CRPS diff {crps_diff:.4f} > {TOLERANCE}"


def test_e2e_exact_nll(setup):
    """
    Exact analytic NLL via teacher_forced_nll(), compared against HF's own
    training-mode loss (HF returns .loss when future_values is passed to
    forward() — this is HF's real NLL, not a sample-based proxy on either side).
    Replaces the Normal-fit-on-samples proxy flagged in review.
    """
    device, hf_model, weights, inputs = setup

    past_values = inputs["input_past_values"]
    past_time = inputs["input_past_time_features"]
    future_time = inputs["input_future_time_features"]
    future_values = inputs["input_future_values"]
    past_mask = inputs["input_past_observed_mask"]
    static_cat = inputs["input_static_categorical_features"].long()
    static_real = inputs["input_static_real_features"]

    with torch.no_grad():
        hf_out = hf_model(
            past_values=past_values,
            past_time_features=past_time,
            future_time_features=future_time,
            past_observed_mask=past_mask,
            static_categorical_features=static_cat,
            static_real_features=static_real,
            future_values=future_values,
            future_observed_mask=torch.ones_like(future_values),
        )
    hf_nll = hf_out.loss.item()
    logger.info(f"HF exact NLL (forward().loss) = {hf_nll:.4f}")

    tt_nll = teacher_forced_nll(
        device=device,
        weights=weights,
        past_values=past_values,
        past_time_features=past_time,
        future_time_features=future_time,
        future_values=future_values,
        past_observed_mask=past_mask,
        static_categorical_features=static_cat,
        static_real_features=static_real,
    )
    logger.info(f"TTNN exact NLL (teacher_forced_nll) = {tt_nll:.4f}")

    nll_diff = abs(tt_nll - hf_nll) / (abs(hf_nll) + 1e-8)
    logger.info(f"NLL relative diff: {nll_diff:.4f} (threshold {TOLERANCE})")
    assert nll_diff <= TOLERANCE, f"NLL diff {nll_diff:.4f} > {TOLERANCE}"
    logger.info("PASSED: exact NLL within 5% of HF reference")


def mean_prediction_mae(samples, targets):
    """
    Point-forecast MAE: mean the sample draws down to a single point forecast
    per (batch, timestep), then compare against the real future values.

    samples: [B, S, T] (S = num_parallel_samples draws from the predicted
             distribution). targets: [B, T].
    Returns a scalar float.
    """
    point_forecast = samples.mean(dim=1)  # [B, T]
    return (point_forecast - targets).abs().mean().item()


@pytest.mark.timeout(600)
def test_e2e_mean_prediction_mae(setup):
    """
    Bounty Stage 1 acceptance criterion (separate from NLL/CRPS):
    'Mean prediction within 5% MAE of reference.'

    Mirrors test_e2e_generate's structure: generate samples from both HF and
    TTNN on the same real inputs, reduce each to a point forecast via
    sample-mean, compute MAE against the real future_values for each, and
    compare the two MAE figures to each other (not to some absolute
    threshold) -- consistent with how the CRPS/NLL criteria are already
    interpreted in this suite as "TTNN vs HF reference," not "TTNN vs a
    fixed number."
    """
    device, hf_model, weights, inputs = setup

    past_values = inputs["input_past_values"]
    past_time = inputs["input_past_time_features"]
    future_time = inputs["input_future_time_features"]
    past_mask = inputs["input_past_observed_mask"]
    static_cat = inputs["input_static_categorical_features"].long()
    static_real = inputs["input_static_real_features"]
    future_values = inputs["input_future_values"]

    hf_model.config.num_parallel_samples = NUM_SAMPLES
    t0 = time.time()
    hf_out = hf_model.generate(
        past_values=past_values,
        past_time_features=past_time,
        future_time_features=future_time,
        past_observed_mask=past_mask,
        static_categorical_features=static_cat,
        static_real_features=static_real,
    )
    hf_time = time.time() - t0
    hf_samples = hf_out.sequences
    logger.info(f"HF generate (MAE test): {hf_time:.2f}s, samples shape {hf_samples.shape}")

    hf_mae = mean_prediction_mae(hf_samples, future_values)
    logger.info(f"HF mean-prediction MAE={hf_mae:.4f}")

    t0 = time.time()
    tt_samples = generate(
        device=device,
        weights=weights,
        past_values=past_values,
        past_time_features=past_time,
        future_time_features=future_time,
        past_observed_mask=past_mask,
        static_categorical_features=static_cat,
        static_real_features=static_real,
    )
    tt_time = time.time() - t0
    logger.info(f"TTNN generate (MAE test): {tt_time:.2f}s, samples shape {tt_samples.shape}")

    tt_mae = mean_prediction_mae(tt_samples, future_values)
    logger.info(f"TTNN mean-prediction MAE={tt_mae:.4f}")

    mae_diff = abs(tt_mae - hf_mae) / (abs(hf_mae) + 1e-8)
    logger.info(f"Mean-prediction MAE relative diff: {mae_diff:.4f} (threshold {TOLERANCE})")
    assert mae_diff <= TOLERANCE, f"Mean-prediction MAE diff {mae_diff:.4f} > {TOLERANCE}"

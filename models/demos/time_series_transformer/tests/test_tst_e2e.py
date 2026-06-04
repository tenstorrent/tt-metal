# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end validation: TTNN TST generate() vs HF reference.
Checks NLL and CRPS are within 5% of HF reference values.
"""

import pytest
import torch
import ttnn
import time
from pathlib import Path
from safetensors import safe_open
from transformers import TimeSeriesTransformerForPrediction
from loguru import logger
from models.utility_functions import comp_pcc

from tt.tst_model import load_weights, generate

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"
NUM_SAMPLES = 100
TOLERANCE = 0.05   # 5% tolerance on NLL and CRPS


def load_ref(filename):
    tensors = {}
    with safe_open(str(REFERENCE_DIR / filename), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def crps_empirical(samples, targets):
    """
    Empirical CRPS: mean over batch of E[|X-y|] - 0.5*E[|X-X'|]
    samples: [B, S, T], targets: [B, T]
    """
    B, S, T = samples.shape
    y = targets.unsqueeze(1)                         # [B, 1, T]
    term1 = (samples - y).abs().mean(dim=1)          # [B, T]
    # E[|X-X'|] via pairwise: O(S^2) -- use a fast approximation
    s_sorted, _ = samples.sort(dim=1)
    idx = torch.arange(1, S + 1, dtype=torch.float32, device=samples.device)
    term2 = (2 * idx.unsqueeze(0).unsqueeze(-1) - S - 1) * s_sorted
    term2 = term2.sum(dim=1) / (S * S)               # [B, T]
    return (term1 - term2).mean().item()


def nll_from_samples_student_t(samples, targets):
    """Approximate NLL using sample mean/std as proxy (not exact but comparable)."""
    mu    = samples.mean(dim=1)      # [B, T]
    sigma = samples.std(dim=1).clamp_min(1e-6)
    from torch.distributions import Normal
    return -Normal(mu, sigma).log_prob(targets).mean().item()


@pytest.fixture(scope="module")
def setup():
    device = ttnn.open_device(device_id=0)
    hf_model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID).eval()
    weights = load_weights(hf_model, device)
    inputs = load_ref("inputs.safetensors")
    yield device, hf_model, weights, inputs
    ttnn.close_device(device)


def test_e2e_generate(setup):
    """TTNN generate() output NLL and CRPS within 5% of HF reference."""
    device, hf_model, weights, inputs = setup

    past_values     = inputs["input_past_values"]
    past_time       = inputs["input_past_time_features"]
    future_time     = inputs["input_future_time_features"]
    past_mask       = inputs["input_past_observed_mask"]
    static_cat      = inputs["input_static_categorical_features"].long()
    static_real     = inputs["input_static_real_features"]
    future_values   = inputs["input_future_values"]   # ground truth

    # ── HF reference ──────────────────────────────────────────────────────────
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
    hf_samples = hf_out.sequences   # [B, S, T]
    logger.info(f"HF generate: {hf_time:.2f}s, samples shape {hf_samples.shape}")

    hf_crps = crps_empirical(hf_samples, future_values)
    hf_nll  = nll_from_samples_student_t(hf_samples, future_values)
    logger.info(f"HF  CRPS={hf_crps:.4f}  NLL={hf_nll:.4f}")

    # ── TTNN implementation ────────────────────────────────────────────────────
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
        num_parallel_samples=NUM_SAMPLES,
    )
    tt_time = time.time() - t0
    logger.info(f"TTNN generate: {tt_time:.2f}s, samples shape {tt_samples.shape}")

    tt_crps = crps_empirical(tt_samples, future_values)
    tt_nll  = nll_from_samples_student_t(tt_samples, future_values)
    logger.info(f"TTNN CRPS={tt_crps:.4f}  NLL={tt_nll:.4f}")

    # ── Tolerance checks ──────────────────────────────────────────────────────
    crps_diff = abs(tt_crps - hf_crps) / (abs(hf_crps) + 1e-8)
    nll_diff  = abs(tt_nll  - hf_nll)  / (abs(hf_nll)  + 1e-8)
    logger.info(f"CRPS relative diff: {crps_diff:.4f} (threshold {TOLERANCE})")
    logger.info(f"NLL  relative diff: {nll_diff:.4f} (threshold {TOLERANCE})")

    assert crps_diff <= TOLERANCE, f"CRPS diff {crps_diff:.4f} > {TOLERANCE}"
    assert nll_diff  <= TOLERANCE, f"NLL  diff {nll_diff:.4f}  > {TOLERANCE}"
    logger.info("PASSED: NLL and CRPS within 5% of HF reference")

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Zero-shot accuracy test on ETTh1 benchmark dataset.

Requires the ETTh1 dataset downloaded by scripts/prepare_assets.py:
  python scripts/prepare_assets.py --datasets etthi

This test is marked ``slow`` and is excluded from CI unless explicitly
requested with ``pytest -m slow``.

Published ETTh1 zero-shot results for TTM-R1 (forecast_length=96):
  MSE ≈ 0.444, MAE ≈ 0.440  (from the TTM paper, Table 1)

These are reported on **normalized** data (each channel zero-mean / unit-std
using train-set statistics) on the standard test split:
  Train: rows 0–8544 (8545 rows)
  Val:   rows 8545–11425 (2881 rows)
  Test:  rows 11426–17419 (5994 rows)

The TTNN model is validated against:
  1. The PyTorch reference (PCC ≥ 0.99 per window, across all 7 channels).
  2. Published MSE within 5% (Stage 3 target).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.demos.granite_ttm_r1.common import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_FORECAST_LENGTH,
    DEFAULT_MODEL_NAME,
    load_granite_ttm_config,
    load_granite_ttm_reference_model,
)
from models.demos.granite_ttm_r1.reference.eval import mae, mse, pcc
from models.demos.granite_ttm_r1.reference.model import extract_prediction_tensor
from models.demos.granite_ttm_r1.reference.preprocess import build_reference_inputs, sliding_windows
from models.demos.granite_ttm_r1.tt.common import preprocess_inputs, preprocess_parameters, to_torch_tensor
from models.demos.granite_ttm_r1.tt.config import GraniteTTMModelConfig
from models.demos.granite_ttm_r1.tt.ttnn_granite_ttm_model import TtnnGraniteTTMModel

# Published zero-shot results from the TTM-R1 paper (ETTh1, 96-step forecast,
# multivariate, normalized with train-set statistics).
PUBLISHED_MSE = 0.444
PUBLISHED_MAE = 0.440
TOLERANCE = 0.05  # Stage 3 target: within 5% of published

# ETTh1 standard train/val/test split (rows, not timestamps)
N_TRAIN = 8545
N_VAL = 2881

DATA_DIR = Path(__file__).parent.parent.parent / "data"
ETTHI_CSV = DATA_DIR / "etthi.csv"

MIN_WINDOWS = 20  # minimum evaluation windows required for a valid result


@pytest.mark.slow
@pytest.mark.skipif(
    not ETTHI_CSV.exists(),
    reason=f"ETTh1 dataset not found at {ETTHI_CSV}. Run: python scripts/prepare_assets.py --datasets etthi",
)
def test_accuracy_etthi(device):
    """Zero-shot multivariate accuracy on ETTh1, within 5% of published TTM-R1 results.

    Uses the standard ETTh1 train/val/test split with per-channel normalization
    from the train set.  All 7 channels are evaluated simultaneously.
    """
    import pandas as pd

    df = pd.read_csv(ETTHI_CSV, index_col=0, parse_dates=True)
    assert (
        len(df) >= N_TRAIN + N_VAL + DEFAULT_CONTEXT_LENGTH + DEFAULT_FORECAST_LENGTH
    ), f"ETTh1 CSV too short: {len(df)} rows"

    # Compute train-set mean/std per channel for normalization
    train_values = torch.tensor(df.values[:N_TRAIN], dtype=torch.float32)  # [N_TRAIN, 7]
    train_mean = train_values.mean(0, keepdim=True)  # [1, 7]
    train_std = train_values.std(0, keepdim=True)  # [1, 7]

    # Normalize the full series with train statistics
    series_norm = (torch.tensor(df.values, dtype=torch.float32) - train_mean) / train_std

    # Extract sliding windows from the test split
    n_test_start = N_TRAIN + N_VAL
    histories, targets = sliding_windows(
        series_norm[n_test_start:],
        context_length=DEFAULT_CONTEXT_LENGTH,
        forecast_length=DEFAULT_FORECAST_LENGTH,
        stride=DEFAULT_FORECAST_LENGTH,
    )
    n_windows = histories.shape[0]
    assert n_windows >= MIN_WINDOWS, f"Need >= {MIN_WINDOWS} test windows, got {n_windows}"
    num_channels = histories.shape[-1]  # 7

    # Build TTNN model
    hf_config = load_granite_ttm_config(DEFAULT_MODEL_NAME)
    model_config = GraniteTTMModelConfig.from_hf_config(hf_config, num_channels=num_channels)

    hf_model = load_granite_ttm_reference_model(DEFAULT_MODEL_NAME, torch_dtype=torch.float32)
    parameters = preprocess_parameters(hf_model, device)

    ttnn_model = TtnnGraniteTTMModel(
        parameters=parameters,
        config=model_config,
        reference_model=hf_model,
    )

    all_preds_ttnn = []
    all_preds_torch = []
    all_targets = []

    for i in range(n_windows):
        history = histories[i : i + 1]  # [1, context_length, 7]
        target = targets[i : i + 1]  # [1, forecast_length, 7]

        # TTNN forward pass
        ttnn_history, ttnn_mask = preprocess_inputs(history, device=device)
        with torch.no_grad():
            ttnn_pred = ttnn_model(ttnn_history, observed_mask=ttnn_mask, device=device)

        # PyTorch reference forward pass (for PCC comparison)
        ref_inputs = build_reference_inputs(hf_model, history)
        with torch.no_grad():
            ref_pred = extract_prediction_tensor(hf_model(**ref_inputs))

        all_preds_ttnn.append(to_torch_tensor(ttnn_pred).float())
        all_preds_torch.append(ref_pred)
        all_targets.append(target)

    preds_ttnn = torch.cat(all_preds_ttnn, dim=0)  # [N, forecast_length, 7]
    preds_torch = torch.cat(all_preds_torch, dim=0)
    actuals = torch.cat(all_targets, dim=0)

    mse_ttnn = float(mse(preds_ttnn, actuals))
    mae_ttnn = float(mae(preds_ttnn, actuals))
    mse_torch = float(mse(preds_torch, actuals))
    pcc_ttnn_vs_torch = float(pcc(preds_ttnn.reshape(-1), preds_torch.reshape(-1)))

    print(f"\nETTh1 zero-shot results ({n_windows} windows, {num_channels} channels, normalized):")
    print(f"  TTNN  MSE: {mse_ttnn:.4f}   MAE: {mae_ttnn:.4f}")
    print(f"  Torch MSE: {mse_torch:.4f}   (reference)")
    print(f"  Published: MSE={PUBLISHED_MSE:.3f}   MAE={PUBLISHED_MAE:.3f}")
    print(f"  PCC (TTNN vs Torch): {pcc_ttnn_vs_torch:.6f}")
    print(f"  Tolerance vs published: {TOLERANCE*100:.0f}%")

    # Gate 1: TTNN predictions closely match PyTorch reference
    assert (
        pcc_ttnn_vs_torch >= 0.99
    ), f"PCC between TTNN and PyTorch {pcc_ttnn_vs_torch:.4f} < 0.99 — TTNN diverges from reference"

    # Gate 2: TTNN MSE is within tolerance of published paper result
    assert mse_ttnn <= PUBLISHED_MSE * (1 + TOLERANCE), (
        f"TTNN MSE {mse_ttnn:.4f} exceeds published {PUBLISHED_MSE:.3f} + {TOLERANCE*100:.0f}%"
        f" (limit: {PUBLISHED_MSE * (1 + TOLERANCE):.4f})"
    )

    # Gate 3: MAE within tolerance
    assert mae_ttnn <= PUBLISHED_MAE * (1 + TOLERANCE), (
        f"TTNN MAE {mae_ttnn:.4f} exceeds published {PUBLISHED_MAE:.3f} + {TOLERANCE*100:.0f}%"
        f" (limit: {PUBLISHED_MAE * (1 + TOLERANCE):.4f})"
    )

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Zero-shot accuracy test on ETTh1 benchmark dataset.

Requires the ETTh1 dataset downloaded by scripts/prepare_assets.py:
  python scripts/prepare_assets.py --datasets etthi

This test is marked ``slow`` and is excluded from CI unless explicitly
requested with ``pytest -m slow``.

Published ETTh1 zero-shot results for TTM-R1 (forecast_length=96):
  MSE ≈ 0.444, MAE ≈ 0.440  (from the TTM paper)

We assert within 10% of these published values.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.demos.granite_ttm_r1.common import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_FORECAST_LENGTH,
    DEFAULT_MODEL_NAME,
    infer_num_channels,
    load_granite_ttm_config,
    load_granite_ttm_reference_model,
)
from models.demos.granite_ttm_r1.reference.eval import mae, mse
from models.demos.granite_ttm_r1.reference.preprocess import sliding_windows
from models.demos.granite_ttm_r1.tt.common import preprocess_inputs, preprocess_parameters, to_torch_tensor
from models.demos.granite_ttm_r1.tt.config import GraniteTTMModelConfig
from models.demos.granite_ttm_r1.tt.ttnn_granite_ttm_model import TtnnGraniteTTMModel

# Published zero-shot results from the TTM-R1 paper (ETTh1, 96-step forecast).
PUBLISHED_MSE = 0.444
PUBLISHED_MAE = 0.440
TOLERANCE = 0.10  # 10% tolerance

DATA_DIR = Path(__file__).parent.parent.parent / "data"
ETTHI_CSV = DATA_DIR / "etthi.csv"

MIN_WINDOWS = 50


@pytest.mark.slow
@pytest.mark.skipif(
    not ETTHI_CSV.exists(),
    reason=f"ETTh1 dataset not found at {ETTHI_CSV}. Run: python scripts/prepare_assets.py --datasets etthi",
)
def test_accuracy_etthi(device):
    """Zero-shot accuracy on ETTh1, within 10% of published TTM-R1 results."""
    import pandas as pd

    df = pd.read_csv(ETTHI_CSV, index_col=0, parse_dates=True)
    # Use OT (oil temperature) column as univariate target, matching TTM-R1 paper setup.
    if "OT" in df.columns:
        series = torch.tensor(df["OT"].values, dtype=torch.float32).unsqueeze(-1)  # [T, 1]
    else:
        series = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32).unsqueeze(-1)

    histories, targets = sliding_windows(
        series,
        context_length=DEFAULT_CONTEXT_LENGTH,
        forecast_length=DEFAULT_FORECAST_LENGTH,
        stride=DEFAULT_FORECAST_LENGTH,
    )
    # histories: [N, context_length, C], targets: [N, forecast_length, C]
    assert histories.shape[0] >= MIN_WINDOWS, f"Need >= {MIN_WINDOWS} windows, got {histories.shape[0]}"

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

    all_predictions = []
    all_targets = []

    for i in range(min(histories.shape[0], 100)):  # cap at 100 windows for speed
        history = histories[i : i + 1]  # [1, context_length, C]
        target = targets[i : i + 1]  # [1, forecast_length, C]

        ttnn_history, ttnn_mask = preprocess_inputs(history, device=device)
        with torch.no_grad():
            prediction = ttnn_model(ttnn_history, observed_mask=ttnn_mask, device=device)

        all_predictions.append(to_torch_tensor(prediction).float())
        all_targets.append(target)

    predictions = torch.cat(all_predictions, dim=0)  # [N, forecast_length, C]
    actuals = torch.cat(all_targets, dim=0)

    mse_val = float(mse(predictions, actuals))
    mae_val = float(mae(predictions, actuals))

    print(f"\nETTh1 zero-shot results ({predictions.shape[0]} windows):")
    print(f"  MSE: {mse_val:.4f}  (published: {PUBLISHED_MSE:.3f}, tolerance: {TOLERANCE*100:.0f}%)")
    print(f"  MAE: {mae_val:.4f}  (published: {PUBLISHED_MAE:.3f}, tolerance: {TOLERANCE*100:.0f}%)")

    assert mse_val <= PUBLISHED_MSE * (
        1 + TOLERANCE
    ), f"MSE {mse_val:.4f} exceeds published {PUBLISHED_MSE:.3f} + {TOLERANCE*100:.0f}%"
    assert mae_val <= PUBLISHED_MAE * (
        1 + TOLERANCE
    ), f"MAE {mae_val:.4f} exceeds published {PUBLISHED_MAE:.3f} + {TOLERANCE*100:.0f}%"

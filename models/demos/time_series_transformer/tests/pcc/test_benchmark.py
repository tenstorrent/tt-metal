# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Forecast quality on the Monash tourism-monthly benchmark.

The parity tests establish that the TTNN model reproduces the HuggingFace reference. This
establishes something different and equally necessary: that the resulting forecasts are
actually good on the dataset the checkpoint was trained for.
"""

import pytest
import torch
from loguru import logger

from models.demos.time_series_transformer.reference.tourism import tourism_series
from models.demos.time_series_transformer.tt.model import TimeSeriesTransformer

BATCH = 8
SAMPLES = 100
QUANTILES = (0.1, 0.5, 0.9)

# A seasonal-naive forecast on tourism-monthly sits around 20-25% MAPE; the published model is
# well inside that. These bounds catch a broken pipeline, not marginal accuracy regressions.
MAX_MAPE_PERCENT = 20.0
MIN_COVERAGE_PERCENT = 60.0
MAX_COVERAGE_PERCENT = 99.5


# Only a genuinely unreachable Hub is grounds for skipping. Schema, parsing and
# feature-construction bugs are regressions in this PR and must fail the benchmark rather
# than disguise themselves as an infrastructure skip.
DOWNLOAD_FAILURES = (
    OSError,  # covers HTTPError, ConnectionError and huggingface_hub's network errors
    TimeoutError,
)


@pytest.fixture(scope="module")
def tourism_inputs(hf_model):
    try:
        return tourism_series(hf_model.config, batch=BATCH)
    except DOWNLOAD_FAILURES as exc:
        pytest.skip(f"tourism-monthly data could not be downloaded: {type(exc).__name__}: {exc}")


class TestTourismBenchmark:
    def test_forecast_quality(self, device, config, hf_state, tourism_inputs):
        inputs = dict(tourism_inputs)
        truth = inputs.pop("future_values")

        model = TimeSeriesTransformer(config, device=device)
        model.load_hf_state_dict(hf_state, strict=True)

        torch.manual_seed(0)
        forecast = model.generate(num_parallel_samples=SAMPLES, mode="sample", **inputs)
        assert forecast.shape == (BATCH, SAMPLES, config.prediction_length)
        assert torch.isfinite(forecast).all()

        median = forecast.median(dim=1).values
        lower = torch.quantile(forecast, QUANTILES[0], dim=1)
        upper = torch.quantile(forecast, QUANTILES[2], dim=1)

        mape = float(torch.mean(torch.abs((median - truth) / truth.clamp_min(1e-6)))) * 100.0
        coverage = float(((truth >= lower) & (truth <= upper)).float().mean()) * 100.0
        logger.info(f"tourism-monthly: MAPE {mape:.2f}%, 80% interval coverage {coverage:.1f}%")

        assert mape < MAX_MAPE_PERCENT, f"median forecast MAPE {mape:.2f}% on tourism-monthly"
        # A degenerate sampler would show up as intervals that cover everything or nothing.
        assert MIN_COVERAGE_PERCENT < coverage < MAX_COVERAGE_PERCENT, f"interval coverage {coverage:.1f}%"

    def test_inputs_are_real_observations(self, tourism_inputs, config):
        """Guard against silently forecasting synthetic data if the loader changes."""
        past = tourism_inputs["past_values"]
        assert past.shape == (BATCH, config.past_length)
        assert torch.isfinite(past).all()
        assert float(past.min()) > 0.0, "tourism counts are positive"
        # Distinct real series, not a repeated template.
        assert len(torch.unique(tourism_inputs["static_categorical_features"])) == BATCH

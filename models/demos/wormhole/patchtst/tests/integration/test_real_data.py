# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from models.demos.wormhole.patchtst.config import PatchTSTDemoConfig, merge_demo_config
from models.demos.wormhole.patchtst.demo.runner import run_patchtst
from models.demos.wormhole.patchtst.reference.hf_reference import reference_forward
from models.demos.wormhole.patchtst.tests.helpers import compute_metrics, prepare_run

PARITY_CORRELATION = 0.90
QUALITY_DELTA_RATIO = 0.05


def _cfg(task: str = "forecast", **overrides) -> PatchTSTDemoConfig:
    defaults = {
        "task": task,
        "strict_fallback": True,
        "dataset": "etth1",
        "batch_size": 1,
        "max_windows": 8,
    }
    defaults.update(overrides)
    return merge_demo_config(PatchTSTDemoConfig(task=defaults["task"]), **defaults)


@pytest.mark.timeout(600)
def test_patchtst_real_data_accuracy_etth1():
    cfg = _cfg(task="forecast")
    prepared = prepare_run(cfg)
    prediction = run_patchtst(cfg)
    reference_prediction = reference_forward(
        artifacts=prepared.reference,
        past_values=prepared.past,
        future_values=prepared.future,
        past_observed_mask=prepared.observed,
    )

    parity_metrics = compute_metrics(prediction, reference_prediction)
    quality_metrics = compute_metrics(prediction, prepared.future)
    reference_quality_metrics = compute_metrics(reference_prediction, prepared.future)
    assert parity_metrics["correlation"] >= PARITY_CORRELATION
    mse_delta = abs(quality_metrics["mse"] - reference_quality_metrics["mse"]) / max(
        abs(reference_quality_metrics["mse"]), 1e-8
    )
    mae_delta = abs(quality_metrics["mae"] - reference_quality_metrics["mae"]) / max(
        abs(reference_quality_metrics["mae"]), 1e-8
    )
    assert mse_delta <= QUALITY_DELTA_RATIO
    assert mae_delta <= QUALITY_DELTA_RATIO
    corr_delta = abs(quality_metrics["correlation"] - reference_quality_metrics["correlation"])
    assert corr_delta <= QUALITY_DELTA_RATIO


@pytest.mark.timeout(600)
def test_patchtst_real_data_weather_sanity():
    cfg = _cfg(task="forecast", dataset="weather", batch_size=4, max_windows=64)
    prepared = prepare_run(cfg)
    prediction = run_patchtst(cfg)
    reference_prediction = reference_forward(
        artifacts=prepared.reference,
        past_values=prepared.past,
        future_values=prepared.future,
        past_observed_mask=prepared.observed,
    )
    parity = compute_metrics(prediction, reference_prediction)
    quality = compute_metrics(prediction, prepared.future)
    reference_quality = compute_metrics(reference_prediction, prepared.future)
    mse_delta = abs(quality["mse"] - reference_quality["mse"]) / max(abs(reference_quality["mse"]), 1e-8)
    mae_delta = abs(quality["mae"] - reference_quality["mae"]) / max(abs(reference_quality["mae"]), 1e-8)
    assert parity["correlation"] >= PARITY_CORRELATION
    assert mse_delta <= QUALITY_DELTA_RATIO
    assert mae_delta <= QUALITY_DELTA_RATIO

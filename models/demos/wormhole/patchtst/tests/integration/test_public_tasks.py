# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from models.demos.wormhole.patchtst.config import PatchTSTDemoConfig, merge_demo_config
from models.demos.wormhole.patchtst.demo.runner import run_patchtst
from models.demos.wormhole.patchtst.reference.hf_reference import reference_forward
from models.demos.wormhole.patchtst.tests.helpers import compute_classification_metrics, compute_metrics, prepare_run

PARITY_CORRELATION = 0.90
PRETRAINING_PARITY_CORRELATION = 0.75
QUALITY_DELTA_RATIO = 0.05


def _cfg(task: str = "forecast", **overrides) -> PatchTSTDemoConfig:
    defaults = {"task": task, "strict_fallback": True, "dataset": "etth1", "batch_size": 1, "max_windows": 64}
    defaults.update(overrides)
    return merge_demo_config(PatchTSTDemoConfig(task=defaults["task"]), **defaults)


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    ("task", "overrides"),
    [
        ("forecast", {}),
        ("forecast", {"channel_mode": "attention", "batch_size": 4, "max_windows": 64}),
        ("regression", {"dataset": "flood_modeling1_reg", "split": "test", "batch_size": 16, "max_windows": 64}),
        (
            "regression",
            {
                "dataset": "flood_modeling1_reg",
                "split": "test",
                "batch_size": 16,
                "max_windows": 64,
                "channel_mode": "attention",
            },
        ),
        ("pretraining", {"batch_size": 4, "max_windows": 64}),
        ("classification", {"dataset": "heartbeat_cls", "split": "test", "batch_size": 16, "max_windows": 64}),
        (
            "classification",
            {
                "dataset": "heartbeat_cls",
                "split": "test",
                "batch_size": 16,
                "max_windows": 64,
                "channel_mode": "attention",
            },
        ),
    ],
)
def test_patchtst_public_task_correctness(task, overrides):
    cfg = _cfg(task=task, **overrides)
    prepared = prepare_run(cfg)
    prediction = run_patchtst(cfg)
    reference_prediction = reference_forward(
        artifacts=prepared.reference,
        past_values=prepared.past,
        future_values=prepared.future,
        target_values=prepared.target_values,
        past_observed_mask=prepared.observed,
    )

    parity = compute_metrics(prediction, reference_prediction)
    assert parity["correlation"] >= (PRETRAINING_PARITY_CORRELATION if task == "pretraining" else PARITY_CORRELATION)

    if task == "classification":
        metrics = compute_classification_metrics(prediction, prepared.target_values)
        reference_metrics = compute_classification_metrics(reference_prediction, prepared.target_values)
        assert abs(metrics["accuracy"] - reference_metrics["accuracy"]) <= 1e-6
        assert abs(metrics["f1_macro"] - reference_metrics["f1_macro"]) <= 1e-6
    elif task == "regression":
        quality = compute_metrics(prediction, prepared.target_values)
        reference_quality = compute_metrics(reference_prediction, prepared.target_values)
        mse_delta = abs(quality["mse"] - reference_quality["mse"]) / max(reference_quality["mse"], 1e-8)
        mae_delta = abs(quality["mae"] - reference_quality["mae"]) / max(reference_quality["mae"], 1e-8)
        assert mse_delta <= QUALITY_DELTA_RATIO
        assert mae_delta <= QUALITY_DELTA_RATIO


@pytest.mark.timeout(600)
def test_patchtst_native_long_context():
    cfg = _cfg(
        task="forecast",
        split="train",
        batch_size=4,
        max_windows=16,
        context_length=4096,
        prediction_length=96,
        checkpoint_id_override="models/demos/wormhole/patchtst/artifacts/finetune/forecast_long_context_etth1_4096_ckpt",
        checkpoint_revision_override="local-generated",
        allow_reference_context_adaptation=False,
        allow_reference_channel_adaptation=False,
    )
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
    mse_delta = abs(quality["mse"] - reference_quality["mse"]) / max(reference_quality["mse"], 1e-8)
    mae_delta = abs(quality["mae"] - reference_quality["mae"]) / max(reference_quality["mae"], 1e-8)
    assert parity["correlation"] >= PARITY_CORRELATION
    assert mse_delta <= QUALITY_DELTA_RATIO
    assert mae_delta <= QUALITY_DELTA_RATIO


@pytest.mark.timeout(600)
def test_patchtst_native_high_channel():
    cfg = _cfg(
        task="forecast",
        dataset="traffic",
        batch_size=4,
        max_windows=16,
        checkpoint_id_override="models/demos/wormhole/patchtst/artifacts/finetune/forecast_high_channel_traffic_ckpt",
        checkpoint_revision_override="local-generated",
        allow_reference_context_adaptation=False,
        allow_reference_channel_adaptation=False,
    )
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
    mse_delta = abs(quality["mse"] - reference_quality["mse"]) / max(reference_quality["mse"], 1e-8)
    mae_delta = abs(quality["mae"] - reference_quality["mae"]) / max(reference_quality["mae"], 1e-8)
    assert parity["correlation"] >= PARITY_CORRELATION
    assert mse_delta <= QUALITY_DELTA_RATIO
    assert mae_delta <= QUALITY_DELTA_RATIO


@pytest.mark.timeout(600)
def test_patchtst_public_multi_head_run():
    cfg = _cfg(
        task="multi_task",
        dataset="heartbeat_cls",
        split="test",
        batch_size=4,
        max_windows=8,
        context_length=309,
        prediction_length=96,
    )
    prepared = prepare_run(cfg)
    prediction = run_patchtst(cfg)

    forecast_reference = reference_forward(
        artifacts=prepared.reference,
        past_values=prepared.past,
        future_values=prepared.future,
        past_observed_mask=prepared.observed,
    )
    classification_reference = reference_forward(
        artifacts=prepared.classification_reference,
        past_values=prepared.past,
        target_values=prepared.target_values,
        past_observed_mask=prepared.observed,
    )
    forecast_parity = compute_metrics(prediction["forecast"], forecast_reference)
    forecast_quality = compute_metrics(prediction["forecast"], prepared.future)
    reference_forecast_quality = compute_metrics(forecast_reference, prepared.future)
    classification_metrics = compute_classification_metrics(prediction["classification"], prepared.target_values)
    reference_classification_metrics = compute_classification_metrics(classification_reference, prepared.target_values)
    mse_delta = abs(forecast_quality["mse"] - reference_forecast_quality["mse"]) / max(
        reference_forecast_quality["mse"], 1e-8
    )
    mae_delta = abs(forecast_quality["mae"] - reference_forecast_quality["mae"]) / max(
        reference_forecast_quality["mae"], 1e-8
    )
    assert forecast_parity["correlation"] >= PARITY_CORRELATION
    assert mse_delta <= QUALITY_DELTA_RATIO
    assert mae_delta <= QUALITY_DELTA_RATIO
    assert abs(classification_metrics["accuracy"] - reference_classification_metrics["accuracy"]) <= 1e-6
    assert abs(classification_metrics["f1_macro"] - reference_classification_metrics["f1_macro"]) <= 1e-6

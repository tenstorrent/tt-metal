# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from loguru import logger

from models.demos.granite_ttm_r1.common import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_FORECAST_LENGTH,
    DEFAULT_MODEL_NAME,
    create_synthetic_example,
    infer_num_channels,
    load_granite_ttm_config,
)
from models.demos.granite_ttm_r1.reference.eval import summarize_regression_metrics
from models.demos.granite_ttm_r1.reference.model import GraniteTTMReferenceModel
from models.demos.granite_ttm_r1.ttnn.common import preprocess_inputs
from models.demos.granite_ttm_r1.ttnn.ttnn_granite_ttm_model import TtnnGraniteTTMModel


def run_granite_ttm_reference_demo(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    forecast_length: int = DEFAULT_FORECAST_LENGTH,
    batch_size: int = 1,
    num_channels: int | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    device=None,
):
    config = load_granite_ttm_config(model_name=model_name, cache_dir=cache_dir)
    resolved_num_channels = num_channels or infer_num_channels(config)
    example = create_synthetic_example(
        batch_size=batch_size,
        context_length=context_length,
        forecast_length=forecast_length,
        num_channels=resolved_num_channels,
    )

    reference = GraniteTTMReferenceModel.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    reference_prediction = reference.predict(example.history, future_values=example.future_values)
    logger.info("Reference output shape: {}", tuple(reference_prediction.shape))

    if device is None:
        return {
            "config": config,
            "reference_prediction": reference_prediction,
            "metrics": summarize_regression_metrics(reference_prediction, reference_prediction),
        }

    ttnn_history, ttnn_mask = preprocess_inputs(example.history, example.observed_mask, device=device)
    model = TtnnGraniteTTMModel(config=config, reference_model=reference.model)
    ttnn_prediction = model(
        ttnn_history,
        observed_mask=ttnn_mask,
        future_values=example.future_values,
        device=device,
    )
    logger.info("TTNN fallback output shape: {}", tuple(ttnn_prediction.shape))

    return {
        "config": config,
        "reference_prediction": reference_prediction,
        "ttnn_prediction": ttnn_prediction,
    }

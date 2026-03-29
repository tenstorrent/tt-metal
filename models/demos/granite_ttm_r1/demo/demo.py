# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

import torch
from loguru import logger

from models.demos.granite_ttm_r1.common import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_FORECAST_LENGTH,
    DEFAULT_MODEL_NAME,
    create_synthetic_example,
    infer_num_channels,
    load_granite_ttm_config,
    load_granite_ttm_reference_model,
)
from models.demos.granite_ttm_r1.reference.eval import pcc, summarize_regression_metrics
from models.demos.granite_ttm_r1.reference.model import extract_prediction_tensor
from models.demos.granite_ttm_r1.tt.common import preprocess_inputs, preprocess_parameters, to_torch_tensor
from models.demos.granite_ttm_r1.tt.config import GraniteTTMModelConfig
from models.demos.granite_ttm_r1.tt.ttnn_granite_ttm_model import TtnnGraniteTTMModel


def run_granite_ttm_demo(
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
    hf_config = load_granite_ttm_config(model_name=model_name, cache_dir=cache_dir)
    resolved_num_channels = num_channels or infer_num_channels(hf_config)
    model_config = GraniteTTMModelConfig.from_hf_config(hf_config, num_channels=resolved_num_channels)

    example = create_synthetic_example(
        batch_size=batch_size,
        context_length=context_length,
        forecast_length=forecast_length,
        num_channels=resolved_num_channels,
    )

    # Load reference model in float32 for accurate comparison.
    hf_model = load_granite_ttm_reference_model(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        torch_dtype=torch.float32,
    )
    with torch.no_grad():
        from models.demos.granite_ttm_r1.reference.preprocess import build_reference_inputs

        ref_inputs = build_reference_inputs(
            hf_model, example.history, future_values=example.future_values, observed_mask=example.observed_mask
        )
        ref_outputs = hf_model(**ref_inputs)
    reference_prediction = extract_prediction_tensor(ref_outputs)
    logger.info("Reference output shape: {}", tuple(reference_prediction.shape))

    if device is None:
        return {
            "hf_config": hf_config,
            "model_config": model_config,
            "reference_prediction": reference_prediction,
            "metrics": summarize_regression_metrics(reference_prediction, reference_prediction),
        }

    # Pre-process weights for TTNN.
    parameters = preprocess_parameters(hf_model, device)

    # Build the TTNN model.
    ttnn_model = TtnnGraniteTTMModel(
        parameters=parameters,
        config=model_config,
        reference_model=hf_model,
    )

    ttnn_history, ttnn_mask = preprocess_inputs(example.history, example.observed_mask, device=device)

    # Warm-up pass.
    with torch.no_grad():
        _ = ttnn_model(ttnn_history, observed_mask=ttnn_mask, device=device)

    # Timed pass.
    N_TIMING = 20
    t0 = time.perf_counter()
    for _ in range(N_TIMING):
        ttnn_prediction = ttnn_model(ttnn_history, observed_mask=ttnn_mask, device=device)
    elapsed = time.perf_counter() - t0
    latency_ms = elapsed / N_TIMING * 1000
    throughput = batch_size * N_TIMING / elapsed

    torch_prediction = to_torch_tensor(ttnn_prediction).float()
    pcc_val = float(pcc(torch_prediction, reference_prediction))
    metrics = summarize_regression_metrics(torch_prediction, reference_prediction)

    logger.info("TTNN output shape: {}", tuple(ttnn_prediction.shape))
    logger.info("PCC vs reference: {:.4f}", pcc_val)
    logger.info("Latency: {:.2f} ms  |  Throughput: {:.1f} seq/s", latency_ms, throughput)

    return {
        "hf_config": hf_config,
        "model_config": model_config,
        "reference_prediction": reference_prediction,
        "ttnn_prediction": ttnn_prediction,
        "pcc": pcc_val,
        "metrics": metrics,
        "latency_ms": latency_ms,
        "throughput_seq_per_s": throughput,
    }


# Backward-compatible alias used by older call-sites.
run_granite_ttm_reference_demo = run_granite_ttm_demo

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Probabilistic forecasting demo for the TTNN Time Series Transformer.

Runs the tourism-monthly checkpoint over a batch of series and reports the median
forecast with prediction intervals, plus calibration coverage against the held-out truth.

    python models/demos/time_series_transformer/demo/demo.py
    python models/demos/time_series_transformer/demo/demo.py --samples 200 --batch 8
"""

from __future__ import annotations

import argparse

import torch
from loguru import logger

import ttnn
from models.demos.time_series_transformer.reference.torch_reference import MODEL_ID
from models.demos.time_series_transformer.reference.tourism import tourism_series
from models.demos.time_series_transformer.tt.model import TimeSeriesTransformer
from models.demos.time_series_transformer.tt.state_io import load_checkpoint_config

TRACE_REGION_SIZE = 64 * 1024 * 1024
QUANTILES = (0.1, 0.5, 0.9)


def synthetic_series(config, *, batch: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """Seasonal series with trend and noise, shaped like monthly tourism data.

    Self-contained so the demo runs with no network access or dataset dependency; swap in
    real observations by replacing ``past_values`` and the time features.
    """
    generator = torch.Generator().manual_seed(seed)
    past_length = int(config.context_length) + int(max(config.lags_sequence))
    total_length = past_length + int(config.prediction_length)

    time_index = torch.arange(total_length, dtype=torch.float32)
    series = []
    for item in range(batch):
        level = 100.0 + 40.0 * item
        seasonal = 20.0 * torch.sin(2.0 * torch.pi * time_index / 12.0 + item)
        trend = 0.4 * time_index
        noise = torch.randn(total_length, generator=generator) * 3.0
        series.append(level + seasonal + trend + noise)
    values = torch.stack(series)

    month_of_year = (time_index % 12.0) / 11.0 - 0.5
    age = time_index / total_length
    time_features = torch.stack((month_of_year, age), dim=-1).expand(batch, total_length, 2)

    return {
        "past_values": values[:, :past_length].contiguous(),
        "future_values": values[:, past_length:].contiguous(),
        "past_time_features": time_features[:, :past_length].contiguous(),
        "future_time_features": time_features[:, past_length:].contiguous(),
        "past_observed_mask": torch.ones(batch, past_length),
        "static_categorical_features": torch.arange(batch).reshape(batch, 1) % int(config.cardinality[0]),
        "static_real_features": torch.zeros(batch, int(config.num_static_real_features)),
    }


def load_series(config, *, batch: int, source: str) -> tuple[dict[str, torch.Tensor], str]:
    if source in ("tourism", "auto"):
        try:
            return tourism_series(config, batch=batch), "tourism_monthly"
        except Exception as exc:  # noqa: BLE001 - offline, or the Hub layout changed
            if source == "tourism":
                raise
            logger.warning(f"falling back to synthetic series ({type(exc).__name__}: {exc})")
    return synthetic_series(config, batch=batch), "synthetic"


def summarize(forecast: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    """Point-forecast error and interval coverage."""
    median = forecast.median(dim=1).values
    lower = torch.quantile(forecast, QUANTILES[0], dim=1)
    upper = torch.quantile(forecast, QUANTILES[2], dim=1)

    inside = ((truth >= lower) & (truth <= upper)).float().mean()
    mae = torch.mean(torch.abs(median - truth))
    mape = torch.mean(torch.abs((median - truth) / truth.clamp_min(1e-6)))
    return {
        "mae": float(mae),
        "mape_percent": float(mape) * 100.0,
        "coverage_80_percent": float(inside) * 100.0,
        "mean_interval_width": float((upper - lower).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=MODEL_ID, help="Hub id or local checkpoint directory")
    parser.add_argument("--batch", type=int, default=4, help="number of series to forecast")
    parser.add_argument("--samples", type=int, default=100, help="trajectories drawn per series")
    parser.add_argument("--series", type=int, default=0, help="which series to print in detail")
    parser.add_argument(
        "--data",
        choices=("auto", "tourism", "synthetic"),
        default="auto",
        help="'tourism' fetches real Monash data from the Hub; 'auto' falls back to synthetic",
    )
    parser.add_argument(
        "--profile",
        choices=("accuracy", "performance"),
        default="accuracy",
        help="accuracy: float32 eager attention; performance: bfloat16 with the flash kernel",
    )
    args = parser.parse_args()

    hf_config = load_checkpoint_config(args.checkpoint)
    inputs, source = load_series(hf_config, batch=args.batch, source=args.data)
    truth = inputs.pop("future_values")
    logger.info(f"forecasting {args.batch} series from {source}")

    overrides = (
        {"dtype": "float32"}
        if args.profile == "accuracy"
        else {"dtype": "bfloat16", "use_sdpa": True, "use_exact_softmax": False}
    )

    device = ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION_SIZE)
    try:
        model = TimeSeriesTransformer.from_pretrained(args.checkpoint, device=device, use_trace=True, **overrides)
        logger.info(
            f"loaded {args.checkpoint}: d_model={model.config.d_model} "
            f"context={model.config.context_length} horizon={model.config.prediction_length} "
            f"distribution={model.config.distribution_output}"
        )

        forecast = model.generate(num_parallel_samples=args.samples, mode="sample", **inputs)
        logger.info(f"drew {args.samples} trajectories for {args.batch} series: {tuple(forecast.shape)}")

        metrics = summarize(forecast, truth)
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.3f}")

        index = args.series
        quantiles = torch.quantile(forecast[index], torch.tensor(QUANTILES, dtype=forecast.dtype), dim=0)
        logger.info(f"\nseries {index}: horizon forecast vs truth")
        logger.info(f"{'step':>5} {'p10':>10} {'p50':>10} {'p90':>10} {'truth':>10}")
        for step in range(model.config.prediction_length):
            logger.info(
                f"{step:>5} {quantiles[0, step]:>10.2f} {quantiles[1, step]:>10.2f} "
                f"{quantiles[2, step]:>10.2f} {truth[index, step]:>10.2f}"
            )

        model.release_traces()
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()

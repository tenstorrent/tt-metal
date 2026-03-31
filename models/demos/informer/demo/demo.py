# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Informer TTNN HF demo (dataset + weights).

Examples:
  python models/demos/informer/demo/demo.py \
    --hf-model-id huggingface/informer-tourism-monthly \
    --hf-dataset monash_tsf --hf-dataset-config tourism_monthly \
    --hf-split test --hf-freq M --hf-max-series 128

  python models/demos/informer/demo/demo.py \
    --hf-model-id huggingface/informer-tourism-monthly \
    --csv /path/to/ETTh1.csv --csv-freq H
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.informer.reference.torch_reference import compute_metrics, load_etth1_csv
from models.demos.informer.tt.config import InformerConfig, informer_config_from_hf
from models.demos.informer.tt.model import create_informer


def load_checkpoint(path: str | None) -> tuple[dict | None, dict | None, torch.Tensor | None, torch.Tensor | None]:
    if not path:
        return None, None, None, None
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Expected checkpoint dict payload, got {type(state).__name__}")
    return state, state.get("config"), state.get("mean"), state.get("std")


def normalize_time_feature_freq(freq: str) -> str:
    if freq == "M":
        return "ME"
    if freq == "Q":
        return "QE"
    if freq == "Y":
        return "YE"
    if freq == "H":
        return "h"
    return freq


def build_time_features_from_start(start, *, length: int, dim: int, freq: str) -> torch.Tensor:
    if dim <= 0:
        return torch.zeros((length, 0), dtype=torch.float32)
    import pandas as pd

    start_ts = pd.Timestamp(start)
    normalized_freq = normalize_time_feature_freq(freq)
    dates = pd.date_range(start=start_ts, periods=length, freq=normalized_freq)
    import numpy as np
    from gluonts.time_feature import time_features_from_frequency_str

    time_feats = time_features_from_frequency_str(normalized_freq)
    features = [feat(dates) for feat in time_feats]
    if len(features) < dim:
        age = np.log10(2.0 + np.arange(length, dtype=np.float32))
        features.append(age)
    if len(features) < dim:
        features.extend([np.zeros(length, dtype=np.float32) for _ in range(dim - len(features))])
    stacked = np.stack(features[:dim], axis=1)
    return torch.tensor(stacked, dtype=torch.float32)


def open_device(*, device_id: int) -> ttnn.Device:
    return ttnn.open_device(device_id=device_id, trace_region_size=128 << 20)


def load_series_from_csv(path: Path, *, cfg: InformerConfig) -> tuple[list[dict], str]:
    timestamps, values = load_etth1_csv(path, features=cfg.enc_in)
    if values.shape[0] < cfg.seq_len + cfg.pred_len:
        raise ValueError("CSV series is shorter than seq_len + pred_len.")
    start = timestamps.iloc[0] if timestamps is not None else 0
    series = [
        {
            "target": values.numpy(),
            "start": start,
            "feat_static_cat": [],
            "feat_static_real": [],
        }
    ]
    return series, "csv"


def run_hf_demo(args: argparse.Namespace, *, device: ttnn.Device | None = None) -> None:
    from datasets import load_dataset
    from transformers import InformerConfig as HfInformerConfig
    from transformers import InformerForPrediction

    if not args.hf_model_id:
        raise ValueError("Provide --hf-model-id to load model weights.")

    logger.info("HF model id: {}", args.hf_model_id)
    logger.info("Generation mode: {}", args.generation_mode)
    hf_config = HfInformerConfig.from_pretrained(args.hf_model_id)
    hf_model = InformerForPrediction.from_pretrained(args.hf_model_id)
    hf_model.eval()
    hf_model.config.num_parallel_samples = args.hf_num_samples

    cfg = informer_config_from_hf(
        hf_config,
        device_id=args.device_id,
        dtype=args.dtype,
        hf_mask_value=args.hf_mask_value,
        hf_compute_dtype=args.hf_compute_dtype,
    )

    if args.csv:
        logger.info("Using CSV dataset: {}", args.csv)
        series, series_kind = load_series_from_csv(Path(args.csv), cfg=cfg)
        freq = args.csv_freq
    else:
        if not args.hf_dataset:
            raise ValueError("Provide --hf-dataset/--hf-dataset-config or --csv to evaluate.")
        logger.info(
            "Using HF dataset: {} (config={}, split={})",
            args.hf_dataset,
            args.hf_dataset_config,
            args.hf_split,
        )
        dataset = load_dataset(args.hf_dataset, args.hf_dataset_config, split=args.hf_split)
        if args.hf_max_series is not None and args.hf_max_series > 0:
            dataset = dataset.select(range(min(args.hf_max_series, len(dataset))))
        series = dataset
        series_kind = "hf"
        freq = args.hf_freq

    context_length = int(hf_config.context_length or cfg.seq_len)
    pred_len = int(hf_config.prediction_length or cfg.pred_len)
    max_lag = max(cfg.lags_sequence) if cfg.lags_sequence else 0
    past_length = context_length + int(max_lag)

    own_device = device is None
    if own_device:
        device = open_device(device_id=args.device_id)

    assert device is not None
    try:
        model = create_informer(cfg, device=device, seed=args.seed)
        model.load_hf_state_dict(hf_model.state_dict(), strict=True)

        mse_sum = 0.0
        mae_sum = 0.0
        corr_sum = 0.0
        batches = 0

        batch_targets = []
        batch_past_values = []
        batch_past_time = []
        batch_future_time = []
        batch_static_cat = []
        batch_static_real = []

        def flush_batch() -> None:
            nonlocal mse_sum, mae_sum, corr_sum, batches
            if not batch_past_values:
                return
            past_values = torch.stack(batch_past_values, dim=0)
            past_time = torch.stack(batch_past_time, dim=0)
            future_time = torch.stack(batch_future_time, dim=0)
            static_cat = torch.stack(batch_static_cat, dim=0) if batch_static_cat else None
            static_real = torch.stack(batch_static_real, dim=0) if batch_static_real else None
            future_values = torch.stack(batch_targets, dim=0)

            if args.generation_mode == "sample":
                preds = model.hf_generate(
                    past_values=past_values,
                    past_time_features=past_time,
                    future_time_features=future_time,
                    past_observed_mask=torch.ones_like(past_values),
                    static_categorical_features=static_cat,
                    static_real_features=static_real,
                    num_parallel_samples=args.hf_num_samples,
                )
            elif args.generation_mode == "mean":
                preds = model.hf_generate_mean(
                    past_values=past_values,
                    past_time_features=past_time,
                    future_time_features=future_time,
                    past_observed_mask=torch.ones_like(past_values),
                    static_categorical_features=static_cat,
                    static_real_features=static_real,
                    num_parallel_samples=args.hf_num_samples,
                )
            else:
                raise ValueError(f"Unsupported generation mode: {args.generation_mode}")

            if preds.dim() == 4:
                preds = preds.mean(dim=1)
            preds = preds.squeeze(-1) if preds.shape[-1] == 1 else preds
            future_gt = future_values.squeeze(-1) if future_values.shape[-1] == 1 else future_values

            mse, mae, corr = compute_metrics(preds, future_gt)
            mse_sum += mse
            mae_sum += mae
            corr_sum += corr
            batches += 1

            batch_targets.clear()
            batch_past_values.clear()
            batch_past_time.clear()
            batch_future_time.clear()
            batch_static_cat.clear()
            batch_static_real.clear()

        for item in series:
            target = torch.tensor(item["target"], dtype=torch.float32)
            if target.dim() == 1:
                target = target.unsqueeze(-1)
            if target.shape[0] < past_length + pred_len:
                continue
            window = target[-(past_length + pred_len) :]
            past_values = window[:past_length]
            future_values = window[past_length:]

            time_features = build_time_features_from_start(
                item["start"],
                length=past_length + pred_len,
                dim=cfg.num_time_features,
                freq=freq,
            )
            past_time = time_features[:past_length]
            future_time = time_features[past_length:]

            static_cat = torch.tensor(item.get("feat_static_cat", []), dtype=torch.long)
            if static_cat.numel() == 0:
                static_cat = torch.zeros((cfg.num_static_categorical_features,), dtype=torch.long)
            static_real = torch.tensor(item.get("feat_static_real", []), dtype=torch.float32)
            if static_real.numel() == 0:
                static_real = torch.zeros((cfg.num_static_real_features,), dtype=torch.float32)

            batch_targets.append(future_values)
            batch_past_values.append(past_values)
            batch_past_time.append(past_time)
            batch_future_time.append(future_time)
            batch_static_cat.append(static_cat)
            if cfg.num_static_real_features:
                batch_static_real.append(static_real)

            if len(batch_past_values) >= args.hf_batch:
                flush_batch()

        flush_batch()

        logger.info("HF demo series kind: {}", series_kind)
        logger.info("HF demo series count: {}", len(series))
        logger.info("Batches: {}", batches)
        logger.info("TTNN vs ground truth:")
        logger.info("MSE: {:.6f}", mse_sum / max(1, batches))
        logger.info("MAE: {:.6f}", mae_sum / max(1, batches))
        logger.info("Corr: {:.6f}", corr_sum / max(1, batches))
    finally:
        if own_device:
            ttnn.close_device(device)


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "generation_mode,hf_num_samples,hf_batch",
    [
        ("mean", 1, 16),
    ],
)
def test_demo(device, generation_mode: str, hf_num_samples: int, hf_batch: int) -> None:
    args = argparse.Namespace(
        hf_model_id="huggingface/informer-tourism-monthly",
        hf_dataset="monash_tsf",
        hf_dataset_config="tourism_monthly",
        hf_split="test",
        hf_freq="M",
        hf_batch=hf_batch,
        hf_max_series=32,
        hf_num_samples=hf_num_samples,
        generation_mode=generation_mode,
        csv=None,
        csv_freq="H",
        device_id=0,
        dtype="float32",
        hf_compute_dtype=None,
        hf_mask_value=float(torch.finfo(torch.float32).min),
        seed=42,
    )
    run_hf_demo(args, device=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Informer TTNN HF demo")

    parser.add_argument("--hf-model-id", type=str, default="huggingface/informer-tourism-monthly")
    parser.add_argument("--hf-dataset", type=str, default="monash_tsf")
    parser.add_argument("--hf-dataset-config", type=str, default="tourism_monthly")
    parser.add_argument("--hf-split", type=str, default="test")
    parser.add_argument("--hf-freq", type=str, default="M")
    parser.add_argument("--hf-batch", type=int, default=8)
    parser.add_argument("--hf-max-series", type=int, default=128)
    parser.add_argument("--hf-num-samples", type=int, default=1)
    parser.add_argument(
        "--generation-mode",
        choices=["mean", "sample"],
        default="mean",
        help="Forecasting mode: deterministic mean (default) or stochastic sample",
    )
    parser.add_argument("--csv", type=str, default=None, help="Optional ETTh1-style CSV input")
    parser.add_argument("--csv-freq", type=str, default="H", help="Frequency string for CSV time features")

    parser.add_argument("--device-id", type=int, default=0, help="TT device id")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16", help="TTNN dtype")
    parser.add_argument(
        "--hf-compute-dtype",
        choices=["bfloat16", "float32"],
        default=None,
        help="Optional compute dtype override for HF compatibility path",
    )
    parser.add_argument(
        "--hf-mask-value",
        type=float,
        default=float(torch.finfo(torch.float32).min),
        help="Attention mask value for HF compatibility path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    if args.csv and not Path(args.csv).exists():
        parser.error(f"CSV path does not exist: {args.csv}")
    if not args.csv and not args.hf_dataset:
        parser.error("Provide --csv or --hf-dataset/--hf-dataset-config to evaluate.")
    if not args.hf_model_id:
        parser.error("Provide --hf-model-id to load model weights.")
    return args


def main() -> None:
    args = parse_args()
    run_hf_demo(args)


if __name__ == "__main__":
    main()

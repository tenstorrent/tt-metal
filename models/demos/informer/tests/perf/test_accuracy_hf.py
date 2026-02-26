# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

from models.demos.informer.reference.eval_utils import default_etth1_splits, iter_windows, resolve_eval_range
from models.demos.informer.reference.torch_reference import (
    build_calendar_time_features,
    build_sinusoidal_time_features,
    compute_metrics,
    compute_normalization,
    load_etth1_csv,
    normalize_values,
)
from models.demos.informer.tests.perf.perf_common import (
    ETTH1_CHECKPOINT,
    ETTH1_PATH,
    HF_BATCH,
    HF_DATASET_CONFIG,
    HF_DATASET_ID,
    HF_DATASET_SPLIT,
    HF_DTYPE,
    HF_FREQ,
    HF_MAX_BASELINE_RATIO_MAE,
    HF_MAX_BASELINE_RATIO_MSE,
    HF_MAX_MEAN_BIAS_RATIO,
    HF_MAX_SERIES,
    HF_MAX_STD_RATIO,
    HF_MIN_CORR_VS_GT,
    HF_MODEL_ID,
    HF_NUM_SAMPLES,
    HF_REF_DTYPE,
    REALDATA_MAX_WINDOWS,
    REALDATA_MIN_CORR,
    REALDATA_MIN_IMPROVEMENT,
    REALDATA_SPLIT,
    REALDATA_STRIDE,
    REALDATA_TIME_FEATURES,
    TARGET_CORRELATION,
    build_config_from_checkpoint,
    build_time_features_from_start,
    hf_generate_mean_reference,
    load_checkpoint,
)
from models.demos.informer.tt.config import informer_config_from_hf
from models.demos.informer.tt.model import InformerModel
from models.demos.informer.tt.ops import to_torch


class TestInformerAccuracyHF:
    """Real-data and HF-checkpoint accuracy validation."""

    @pytest.mark.models_performance_bare_metal
    def test_real_dataset_accuracy(self, device):
        dataset_path = ETTH1_PATH
        checkpoint_path = ETTH1_CHECKPOINT
        if not dataset_path.is_file() or not checkpoint_path.is_file():
            pytest.skip(f"Missing required assets for real-data accuracy check: {dataset_path} and {checkpoint_path}.")

        state, cfg_override, mean_ckpt, std_ckpt = load_checkpoint(str(checkpoint_path))
        if state is None or cfg_override is None:
            pytest.skip("Checkpoint missing config or state_dict.")
        ttnn_state = state.get("state_dict")
        torch_state = state.get("torch_state_dict")
        if ttnn_state is None or torch_state is None:
            pytest.skip("Checkpoint missing ttnn or torch weights.")

        cfg = build_config_from_checkpoint(cfg_override)
        from models.demos.informer.reference.torch_reference import TorchInformerModel

        torch_model = TorchInformerModel(cfg)
        torch_model.load_state_dict(torch_state, strict=True)
        torch_model.eval()

        ttnn_model = InformerModel(cfg, device=device, seed=42)
        ttnn_model.load_state_dict(ttnn_state, strict=True)
        timestamps, values = load_etth1_csv(dataset_path, features=cfg.enc_in)
        if values.shape[0] < cfg.seq_len + cfg.pred_len:
            pytest.skip("Dataset shorter than seq_len + pred_len.")

        if REALDATA_TIME_FEATURES == "sin":
            time_features = build_sinusoidal_time_features(values.shape[0], cfg.time_feature_dim)
        else:
            time_features = build_calendar_time_features(timestamps, values.shape[0], cfg.time_feature_dim)

        mean = mean_ckpt
        std = std_ckpt
        if mean is None or std is None:
            split_cfg = default_etth1_splits()
            mean, std = compute_normalization(values, min(split_cfg.train_len, values.shape[0]))
        values = normalize_values(values, mean, std)

        split_cfg = default_etth1_splits()
        eval_start, eval_end = resolve_eval_range(
            values.shape[0],
            split=REALDATA_SPLIT,
            split_cfg=split_cfg,
        )
        eval_length = eval_end - eval_start

        mse_ttnn = []
        mae_ttnn = []
        corr_ttnn = []
        mse_torch = []
        mae_torch = []
        corr_torch = []
        corr_vs_ref = []
        mse_baseline = []
        mae_baseline = []
        corr_baseline = []

        for offset in iter_windows(
            eval_length,
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
            stride=REALDATA_STRIDE,
            max_windows=REALDATA_MAX_WINDOWS,
        ):
            start = eval_start + offset
            window = values[start : start + cfg.seq_len + cfg.pred_len]
            past_values = window[: cfg.seq_len].unsqueeze(0)
            future_values = window[cfg.seq_len :].unsqueeze(0)

            past_time = time_features[start : start + cfg.seq_len].unsqueeze(0)
            future_time = time_features[start + cfg.seq_len : start + cfg.seq_len + cfg.pred_len].unsqueeze(0)

            with torch.no_grad():
                torch_out = torch_model(past_values, past_time, future_time, future_values)
            ttnn_out = to_torch(ttnn_model(past_values, past_time, future_time, future_values=future_values))
            baseline = past_values[:, -1:, :].repeat(1, cfg.pred_len, 1)

            mse, mae, corr = compute_metrics(ttnn_out, future_values)
            mse_ref, mae_ref, corr_ref = compute_metrics(torch_out, future_values)
            _, _, corr_tt_ref = compute_metrics(ttnn_out, torch_out)
            mse_b, mae_b, corr_b = compute_metrics(baseline, future_values)

            mse_ttnn.append(mse)
            mae_ttnn.append(mae)
            corr_ttnn.append(corr)
            mse_torch.append(mse_ref)
            mae_torch.append(mae_ref)
            corr_torch.append(corr_ref)
            corr_vs_ref.append(corr_tt_ref)
            mse_baseline.append(mse_b)
            mae_baseline.append(mae_b)
            corr_baseline.append(corr_b)

        avg_mse = sum(mse_ttnn) / len(mse_ttnn)
        avg_mae = sum(mae_ttnn) / len(mae_ttnn)
        avg_corr = sum(corr_ttnn) / len(corr_ttnn)
        avg_mse_ref = sum(mse_torch) / len(mse_torch)
        avg_mae_ref = sum(mae_torch) / len(mae_torch)
        avg_corr_ref = sum(corr_torch) / len(corr_torch)
        avg_corr_vs_ref = sum(corr_vs_ref) / len(corr_vs_ref)
        avg_mse_baseline = sum(mse_baseline) / len(mse_baseline)
        avg_mae_baseline = sum(mae_baseline) / len(mae_baseline)
        avg_corr_baseline = sum(corr_baseline) / len(corr_baseline)

        logger.info(
            "Real-data (ETTh1) TTNN vs GT: MSE {:.6f}, MAE {:.6f}, Corr {:.4f}",
            avg_mse,
            avg_mae,
            avg_corr,
        )
        logger.info(
            "Real-data (ETTh1) Torch vs GT: MSE {:.6f}, MAE {:.6f}, Corr {:.4f}",
            avg_mse_ref,
            avg_mae_ref,
            avg_corr_ref,
        )
        logger.info("Real-data (ETTh1) TTNN vs Torch Corr: {:.4f}", avg_corr_vs_ref)
        logger.info(
            "Real-data (ETTh1) baseline vs GT: MSE {:.6f}, MAE {:.6f}, Corr {:.4f}",
            avg_mse_baseline,
            avg_mae_baseline,
            avg_corr_baseline,
        )

        mse_target = (1.0 - REALDATA_MIN_IMPROVEMENT) * avg_mse_baseline
        mae_target = (1.0 - REALDATA_MIN_IMPROVEMENT) * avg_mae_baseline
        assert avg_mse_ref <= mse_target
        assert avg_mae_ref <= mae_target
        assert avg_corr_ref >= REALDATA_MIN_CORR
        assert avg_mse <= mse_target
        assert avg_mae <= mae_target
        assert avg_corr >= REALDATA_MIN_CORR

        assert abs(avg_mse - avg_mse_ref) <= 0.05 * max(avg_mse_ref, 1e-8)
        assert abs(avg_mae - avg_mae_ref) <= 0.05 * max(avg_mae_ref, 1e-8)
        assert avg_corr_vs_ref > TARGET_CORRELATION

    @pytest.mark.models_performance_bare_metal
    def test_hf_checkpoint_accuracy(self, device):
        torch.manual_seed(0)
        model_id = HF_MODEL_ID

        from datasets import load_dataset
        from transformers import InformerConfig as HfInformerConfig
        from transformers import InformerForPrediction

        dataset_id = HF_DATASET_ID
        dataset_config = HF_DATASET_CONFIG
        dataset_split = HF_DATASET_SPLIT
        freq = HF_FREQ
        max_series = HF_MAX_SERIES
        batch_size = HF_BATCH
        num_samples = HF_NUM_SAMPLES
        hf_config = HfInformerConfig.from_pretrained(model_id)
        hf_model = InformerForPrediction.from_pretrained(model_id)
        hf_model.eval()
        ref_dtype = HF_REF_DTYPE
        if ref_dtype == "bfloat16":
            hf_model = hf_model.to(torch.bfloat16)
        elif ref_dtype == "float16":
            hf_model = hf_model.to(torch.float16)
        elif ref_dtype == "float32":
            hf_model = hf_model.to(torch.float32)
        hf_model.config.num_parallel_samples = num_samples

        dtype = HF_DTYPE
        args = SimpleNamespace(device_id=0, dtype=dtype)
        cfg = informer_config_from_hf(
            hf_config,
            device_id=args.device_id,
            dtype=args.dtype,
        )
        ttnn_model = InformerModel(cfg, device=device, seed=42)
        ttnn_model.load_hf_state_dict(hf_model.state_dict(), strict=True)

        dataset = load_dataset(dataset_id, dataset_config, split=dataset_split)
        if max_series > 0:
            dataset = dataset.select(range(min(max_series, len(dataset))))

        context_length = int(hf_config.context_length or cfg.seq_len)
        pred_len = int(hf_config.prediction_length or cfg.pred_len)
        max_lag = max(cfg.lags_sequence) if cfg.lags_sequence else 0
        past_length = context_length + int(max_lag)

        mse_vals = []
        mae_vals = []
        corr_vals = []
        mse_ref_vals = []
        mae_ref_vals = []
        corr_ref_vals = []
        corr_vs_ref = []
        mse_baseline_vals = []
        mae_baseline_vals = []
        corr_baseline_vals = []
        all_preds = []
        all_future_gt = []
        all_hf_preds = []
        all_baseline_preds = []

        min_corr_vs_gt = HF_MIN_CORR_VS_GT
        max_mse_ratio_vs_baseline = HF_MAX_BASELINE_RATIO_MSE
        max_mae_ratio_vs_baseline = HF_MAX_BASELINE_RATIO_MAE
        max_mean_bias_ratio = HF_MAX_MEAN_BIAS_RATIO
        max_std_ratio = HF_MAX_STD_RATIO

        batch_targets = []
        batch_past_values = []
        batch_past_time = []
        batch_future_time = []
        batch_static_cat = []
        batch_static_real = []

        def flush_batch():
            if not batch_past_values:
                return
            past_values = torch.stack(batch_past_values, dim=0)
            past_time = torch.stack(batch_past_time, dim=0)
            future_time = torch.stack(batch_future_time, dim=0)
            static_cat = torch.stack(batch_static_cat, dim=0) if batch_static_cat else None
            static_real = torch.stack(batch_static_real, dim=0) if batch_static_real else None
            future_values = torch.stack(batch_targets, dim=0)
            baseline = past_values[:, -1:, :].repeat(1, pred_len, 1)
            baseline_eval = baseline.squeeze(-1) if baseline.shape[-1] == 1 else baseline

            rng_state = torch.random.get_rng_state()
            preds = ttnn_model.hf_generate_mean(
                past_values=past_values,
                past_time_features=past_time,
                future_time_features=future_time,
                past_observed_mask=torch.ones_like(past_values),
                static_categorical_features=static_cat,
                static_real_features=static_real,
                num_parallel_samples=num_samples,
            )
            torch.random.set_rng_state(rng_state)

            if preds.dim() == 4:
                preds = preds.mean(dim=1)
            preds = preds.squeeze(-1) if preds.shape[-1] == 1 else preds
            future_gt = future_values.squeeze(-1) if future_values.shape[-1] == 1 else future_values

            mse, mae, corr = compute_metrics(preds, future_gt)
            mse_vals.append(mse)
            mae_vals.append(mae)
            corr_vals.append(corr)
            mse_b, mae_b, corr_b = compute_metrics(baseline_eval, future_gt)
            mse_baseline_vals.append(mse_b)
            mae_baseline_vals.append(mae_b)
            corr_baseline_vals.append(corr_b)

            if ref_dtype == "bfloat16":
                ref_t = torch.bfloat16
            elif ref_dtype == "float16":
                ref_t = torch.float16
            else:
                ref_t = torch.float32
            hf_kwargs = {
                "past_values": past_values.to(dtype=ref_t),
                "past_time_features": past_time.to(dtype=ref_t),
                "future_time_features": future_time.to(dtype=ref_t),
                "past_observed_mask": torch.ones_like(past_values, dtype=ref_t),
                "static_categorical_features": static_cat,
                "static_real_features": static_real.to(dtype=ref_t) if static_real is not None else None,
            }
            with torch.no_grad():
                hf_preds = hf_generate_mean_reference(
                    hf_model,
                    **hf_kwargs,
                    num_parallel_samples=num_samples,
                )
            if not isinstance(hf_preds, torch.Tensor):
                if hasattr(hf_preds, "sequences"):
                    hf_preds = hf_preds.sequences
                elif hasattr(hf_preds, "predictions"):
                    hf_preds = hf_preds.predictions
                elif hasattr(hf_preds, "samples"):
                    hf_preds = hf_preds.samples
            if not isinstance(hf_preds, torch.Tensor):
                raise ValueError("Unsupported HF generate output type.")
            while hf_preds.dim() > 2 and hf_preds.shape[-1] == 1:
                hf_preds = hf_preds.squeeze(-1)
            total = preds.shape[0] * preds.shape[1]
            if hf_preds.numel() == total:
                hf_preds = hf_preds.reshape(preds.shape[0], preds.shape[1])
            elif hf_preds.numel() % total == 0:
                sample_count = hf_preds.numel() // total
                hf_preds = hf_preds.reshape(preds.shape[0], sample_count, preds.shape[1]).mean(dim=1)
            mse_ref, mae_ref, corr_ref_gt = compute_metrics(hf_preds, future_gt)
            mse_ref_vals.append(mse_ref)
            mae_ref_vals.append(mae_ref)
            corr_ref_vals.append(corr_ref_gt)
            _, _, corr_ref = compute_metrics(preds, hf_preds)
            corr_vs_ref.append(corr_ref)

            all_preds.append(preds.reshape(-1))
            all_future_gt.append(future_gt.reshape(-1))
            all_hf_preds.append(hf_preds.reshape(-1))
            all_baseline_preds.append(baseline_eval.reshape(-1))

            batch_targets.clear()
            batch_past_values.clear()
            batch_past_time.clear()
            batch_future_time.clear()
            batch_static_cat.clear()
            batch_static_real.clear()

        for item in dataset:
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

            if len(batch_past_values) >= batch_size:
                flush_batch()

        flush_batch()

        if not mse_vals:
            pytest.skip("No valid HF evaluation windows generated.")

        avg_mse = sum(mse_vals) / len(mse_vals)
        avg_mae = sum(mae_vals) / len(mae_vals)
        avg_corr = sum(corr_vals) / len(corr_vals)
        if not mse_ref_vals:
            pytest.fail("HF reference metrics missing; generate path did not produce comparable outputs.")
        avg_mse_ref = sum(mse_ref_vals) / len(mse_ref_vals)
        avg_mae_ref = sum(mae_ref_vals) / len(mae_ref_vals)
        avg_corr_ref_gt = sum(corr_ref_vals) / len(corr_ref_vals)
        avg_mse_baseline = sum(mse_baseline_vals) / len(mse_baseline_vals)
        avg_mae_baseline = sum(mae_baseline_vals) / len(mae_baseline_vals)
        avg_corr_baseline = sum(corr_baseline_vals) / len(corr_baseline_vals)
        global_preds = torch.cat(all_preds)
        global_future_gt = torch.cat(all_future_gt)
        global_hf_preds = torch.cat(all_hf_preds)
        global_baseline_preds = torch.cat(all_baseline_preds)
        global_mse, global_mae, global_corr = compute_metrics(global_preds, global_future_gt)
        global_mse_ref, global_mae_ref, global_corr_ref_gt = compute_metrics(global_hf_preds, global_future_gt)
        global_mse_baseline, global_mae_baseline, global_corr_baseline = compute_metrics(
            global_baseline_preds, global_future_gt
        )
        _, _, global_corr_ref = compute_metrics(global_preds, global_hf_preds)

        avg_corr_ref = sum(corr_vs_ref) / len(corr_vs_ref)
        global_mean = global_preds.mean().item()
        global_mean_ref = global_hf_preds.mean().item()
        global_std = global_preds.std(unbiased=False).item()
        global_std_ref = global_hf_preds.std(unbiased=False).item()
        mean_bias_ratio = abs(global_mean - global_mean_ref) / max(abs(global_mean_ref), 1e-8)
        std_ratio = global_std / max(global_std_ref, 1e-8)

        mse_ratio_vs_baseline = global_mse / max(global_mse_baseline, 1e-8)
        mae_ratio_vs_baseline = global_mae / max(global_mae_baseline, 1e-8)

        logger.info(
            "HF checkpoint TTNN vs GT: MSE {:.6f}, MAE {:.6f}, Corr {:.4f}",
            avg_mse,
            avg_mae,
            avg_corr,
        )
        logger.info(
            "HF checkpoint HF vs GT: MSE {:.6f}, MAE {:.6f}, Corr {:.4f}",
            avg_mse_ref,
            avg_mae_ref,
            avg_corr_ref_gt,
        )
        logger.info(
            "HF checkpoint baseline vs GT: MSE {:.6f}, MAE {:.6f}, Corr {:.4f}",
            avg_mse_baseline,
            avg_mae_baseline,
            avg_corr_baseline,
        )

        assert mse_ratio_vs_baseline <= max_mse_ratio_vs_baseline
        assert mae_ratio_vs_baseline <= max_mae_ratio_vs_baseline
        assert global_corr >= min_corr_vs_gt
        assert mean_bias_ratio <= max_mean_bias_ratio
        assert (1.0 / max_std_ratio) <= std_ratio <= max_std_ratio
        assert global_corr_ref > TARGET_CORRELATION

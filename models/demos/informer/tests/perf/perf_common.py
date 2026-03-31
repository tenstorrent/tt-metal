# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn
from models.demos.informer.tt.config import InformerConfig
from models.demos.informer.tt.hf_common import (
    torch_negative_binomial_domain_map,
    torch_normal_domain_map,
    torch_student_t_domain_map,
)

# Performance and accuracy targets for this demo.
TARGET_THROUGHPUT = 500  # stretched target: sequences/second for batch inference
TARGET_LATENCY_MS = 20  # stretched target: milliseconds for batch size 1
TARGET_MSE = 0.1
TARGET_MAE = 0.1
TARGET_CORRELATION = 0.90

# Fixed demo datasets/checkpoints (no env tuning)
ASSETS_DIR = Path(__file__).resolve().parents[2] / ".assets"
ETTH1_PATH = ASSETS_DIR / "ETTh1.csv"
ETTH1_CHECKPOINT = ASSETS_DIR / "etth1_ttnn.pt"
HF_MODEL_ID = "huggingface/informer-tourism-monthly"
HF_DATASET_ID = "monash_tsf"
HF_DATASET_CONFIG = "tourism_monthly"
HF_DATASET_SPLIT = "test"
HF_FREQ = "M"
HF_MAX_SERIES = 64
HF_BATCH = 64
HF_NUM_SAMPLES = 1
HF_REF_DTYPE = "float32"
HF_DTYPE = "float32"
HF_MIN_CORR_VS_GT = 0.90
HF_MAX_BASELINE_RATIO_MSE = 2.0
HF_MAX_BASELINE_RATIO_MAE = 2.0
HF_MAX_MEAN_BIAS_RATIO = 1.0
HF_MAX_STD_RATIO = 2.0
REALDATA_TIME_FEATURES = "calendar"
REALDATA_SPLIT = "test"
REALDATA_MAX_WINDOWS = 8
REALDATA_STRIDE = 24
REALDATA_MIN_IMPROVEMENT = 0.05
REALDATA_MIN_CORR = 0.20
LONG_SEQ_MAX_LATENCY_MS = 30000.0
WIDE_MAX_LATENCY_MS = 5000.0


@dataclass
class BenchmarkConfig:
    seq_len: int
    label_len: int
    pred_len: int
    batch: int


def _make_config(**overrides) -> InformerConfig:
    base = dict(
        enc_in=7,
        dec_in=7,
        c_out=7,
        seq_len=96,
        label_len=48,
        pred_len=24,
        d_model=64,
        n_heads=2,
        d_ff=256,
        e_layers=2,
        d_layers=1,
        time_feature_dim=4,
        dtype="bfloat16",
        attention_type="prob",
        use_l1=True,
        use_sharded=True,
        use_sdpa=True,
        use_trace=True,
        use_program_cache=True,
    )
    base.update(overrides)
    return InformerConfig(**base)


def build_config(seq_len: int, pred_len: int, *, label_len: int | None = None) -> InformerConfig:
    if label_len is None:
        label_len = seq_len // 2
    return _make_config(seq_len=seq_len, label_len=label_len, pred_len=pred_len)


def build_config_from_checkpoint(cfg_override: dict) -> InformerConfig:
    def pick(name: str, fallback: int) -> int:
        return int(cfg_override.get(name, fallback))

    return _make_config(
        enc_in=pick("enc_in", 7),
        dec_in=pick("dec_in", 7),
        c_out=pick("c_out", 7),
        seq_len=pick("seq_len", 96),
        label_len=pick("label_len", 48),
        pred_len=pick("pred_len", 24),
        d_model=pick("d_model", 64),
        n_heads=pick("n_heads", 2),
        d_ff=pick("d_ff", 256),
        e_layers=pick("e_layers", 2),
        d_layers=pick("d_layers", 1),
        time_feature_dim=pick("time_feature_dim", 4),
    )


def make_inputs(batch: int, cfg: InformerConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    past_values = torch.randn(batch, cfg.seq_len, cfg.enc_in, dtype=torch.float32)
    past_time = torch.randn(batch, cfg.seq_len, cfg.time_feature_dim, dtype=torch.float32)
    future_time = torch.randn(batch, cfg.pred_len, cfg.time_feature_dim, dtype=torch.float32)
    return past_values, past_time, future_time


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
    import numpy as np
    import pandas as pd
    from gluonts.time_feature import time_features_from_frequency_str

    start_ts = pd.Timestamp(start)
    normalized_freq = normalize_time_feature_freq(freq)
    dates = pd.date_range(start=start_ts, periods=length, freq=normalized_freq)
    time_feats = time_features_from_frequency_str(normalized_freq)
    features = [feat(dates) for feat in time_feats]
    if len(features) < dim:
        age = np.log10(2.0 + np.arange(length, dtype=np.float32))
        features.append(age)
    if len(features) < dim:
        features.extend([np.zeros(length, dtype=np.float32) for _ in range(dim - len(features))])
    stacked = np.stack(features[:dim], axis=1)
    return torch.tensor(stacked, dtype=torch.float32)


def hf_generate_mean_reference(
    hf_model,
    past_values: torch.Tensor,
    past_time_features: torch.Tensor,
    future_time_features: torch.Tensor,
    *,
    past_observed_mask: torch.Tensor,
    static_categorical_features: torch.Tensor | None,
    static_real_features: torch.Tensor | None,
    num_parallel_samples: int,
) -> torch.Tensor:
    # HF Informer expects `[B, T, C]`; univariate inputs may arrive as `[B, T]`.
    if past_values.dim() == 2:
        past_values = past_values.unsqueeze(-1)

    with torch.no_grad():
        outputs = hf_model(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            future_time_features=future_time_features,
            future_values=None,
            return_dict=True,
            use_cache=True,
        )
        decoder = hf_model.model.get_decoder()
        enc_last_hidden = outputs.encoder_last_hidden_state
        loc = outputs.loc
        scale = outputs.scale
        static_feat = outputs.static_features

        repeated_loc = loc.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_scale = scale.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_past_values = (past_values.repeat_interleave(repeats=num_parallel_samples, dim=0) - repeated_loc) / (
            repeated_scale
        )

        expanded_static_feat = static_feat.unsqueeze(1).expand(-1, future_time_features.shape[1], -1)
        features = torch.cat((expanded_static_feat, future_time_features), dim=-1)
        repeated_features = features.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_enc_last_hidden = enc_last_hidden.repeat_interleave(repeats=num_parallel_samples, dim=0)

        future_samples = []
        pred_len = int(hf_model.config.prediction_length)
        dist_name = str(hf_model.config.distribution_output).lower()
        for k in range(pred_len):
            lagged_sequence = hf_model.model.get_lagged_subsequences(
                sequence=repeated_past_values,
                subsequences_length=1 + k,
                shift=1,
            )
            lags_shape = lagged_sequence.shape
            reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)
            decoder_input = torch.cat((reshaped_lagged_sequence, repeated_features[:, : k + 1]), dim=-1)

            dec_output = decoder(inputs_embeds=decoder_input, encoder_hidden_states=repeated_enc_last_hidden)
            params = hf_model.parameter_projection(dec_output.last_hidden_state[:, -1:])
            if dist_name == "student_t":
                df, loc_param, scale_param = torch_student_t_domain_map(params[0], params[1], params[2])
                next_sample = torch.distributions.StudentT(df, loc_param, scale_param).mean
            elif dist_name == "normal":
                loc_param, scale_param = torch_normal_domain_map(
                    params[0],
                    params[1],
                    minimum_scale=float(getattr(hf_model.config, "minimum_scale", 1e-10)),
                )
                next_sample = torch.distributions.Normal(loc_param, scale_param).mean
            elif dist_name == "negative_binomial":
                total_count, logits = torch_negative_binomial_domain_map(
                    params[0],
                    params[1],
                    minimum_scale=float(getattr(hf_model.config, "minimum_scale", 1e-10)),
                )
                next_sample = torch.distributions.NegativeBinomial(total_count=total_count, logits=logits).mean
            else:
                raise ValueError(f"Unsupported distribution_output: {hf_model.config.distribution_output}")
            while next_sample.dim() < repeated_loc.dim():
                next_sample = next_sample.unsqueeze(1)
            next_sample = repeated_loc + repeated_scale * next_sample
            repeated_past_values = torch.cat(
                (repeated_past_values, (next_sample - repeated_loc) / repeated_scale), dim=1
            )
            future_samples.append(next_sample)
        concat_future_samples = torch.cat(future_samples, dim=1)
        batch = past_values.shape[0]
        return concat_future_samples.view(batch, num_parallel_samples, pred_len, -1)


def run_benchmark(
    model,
    past_values,
    past_time,
    future_time,
    *,
    device,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    def call_model():
        return model(past_values, past_time, future_time)

    for _ in range(warmup):
        _ = call_model()
    if device is not None:
        ttnn.synchronize_device(device)
    start = time.time()
    for _ in range(iters):
        _ = call_model()
    if device is not None:
        ttnn.synchronize_device(device)
    elapsed = time.time() - start
    throughput = (past_values.shape[0] * iters) / elapsed
    latency_ms = (elapsed / iters) * 1000.0
    return throughput, latency_ms

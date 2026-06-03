# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

import torch
import torch.nn.functional as F

LAGS = [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 23, 24, 25, 35, 36, 37]
INPUT_SIZE = 1
D_MODEL = 26
NUM_TIME_FEATURES = 2
STATIC_CAT_EMB_DIM = 6
STATIC_REAL_DIM = 1


def _get_lagged_subsequences(values, lags, subsequence_length):
    full_seq = values.shape[1]
    lagged = []
    for lag in lags:
        start = full_seq - lag - subsequence_length
        end   = full_seq - lag
        lagged.append(values[:, start:end].unsqueeze(-1))
    return torch.cat(lagged, dim=-1)


def _mean_scaler(values, mask):
    mask = mask.float()
    sum_vals = (values.abs() * mask).sum(dim=1, keepdim=True)
    count    = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    scale    = (sum_vals / count).clamp(min=1e-6)
    return values / scale, scale


def prepare_encoder_input(
    past_values,
    past_time_features,
    past_observed_mask,
    static_cat_features,
    static_real_features,
    cat_embedder_weight,
    value_proj_weight,
    pos_emb_weight,
):
    B, ctx = past_values.shape
    device = past_values.device

    pv = past_values.unsqueeze(-1).float()
    mask3 = past_observed_mask.unsqueeze(-1).float()
    scaled_pv, scale = _mean_scaler(pv, mask3)

    max_lag = max(LAGS)
    padded = torch.cat([
        torch.zeros(B, max_lag, device=device, dtype=past_values.dtype),
        past_values.float()
    ], dim=1)
    padded_scaled = padded / scale.squeeze(-1)
    lag_feats = _get_lagged_subsequences(padded_scaled, LAGS, ctx)

    cat_emb = F.embedding(static_cat_features[:, 0].long(), cat_embedder_weight)
    cat_emb = cat_emb.unsqueeze(1).expand(-1, ctx, -1)
    stat_real = static_real_features[:, :1].unsqueeze(1).expand(-1, ctx, -1)

    feat = torch.cat([
        scaled_pv,
        lag_feats,
        past_time_features.float(),
        stat_real,
        cat_emb,
        mask3,
    ], dim=-1)

    emb = feat @ value_proj_weight.T
    positions = torch.arange(ctx, device=device)
    pos_emb = pos_emb_weight[positions]
    emb = emb + pos_emb.unsqueeze(0)

    return emb, scale


def prepare_decoder_input(
    future_values,
    future_time_features,
    past_values,
    scale,
    static_cat_features,
    static_real_features,
    cat_embedder_weight,
    value_proj_weight,
    pos_emb_weight,
    context_length=24,
):
    B, pred_len = future_values.shape
    device = future_values.device

    scaled_fv = (future_values.float() / scale.squeeze(-1)).unsqueeze(-1)

    full_seq = torch.cat([past_values.float(), future_values.float()], dim=1)
    max_lag = max(LAGS)
    padded = torch.cat([
        torch.zeros(B, max_lag, device=device, dtype=past_values.dtype),
        full_seq
    ], dim=1)
    padded_scaled = padded / scale.squeeze(-1)
    lag_feats = _get_lagged_subsequences(padded_scaled, LAGS, pred_len)

    cat_emb = F.embedding(static_cat_features[:, 0].long(), cat_embedder_weight)
    cat_emb = cat_emb.unsqueeze(1).expand(-1, pred_len, -1)
    stat_real = static_real_features[:, :1].unsqueeze(1).expand(-1, pred_len, -1)
    obs_mask = torch.ones(B, pred_len, 1, device=device, dtype=torch.float32)

    feat = torch.cat([
        scaled_fv,
        lag_feats,
        future_time_features.float(),
        stat_real,
        cat_emb,
        obs_mask,
    ], dim=-1)

    emb = feat @ value_proj_weight.T
    positions = torch.arange(context_length, context_length + pred_len, device=device)
    pos_emb = pos_emb_weight[positions]
    emb = emb + pos_emb.unsqueeze(0)

    return emb

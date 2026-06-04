# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# Mirrors HuggingFace TimeSeriesTransformerModel.create_network_inputs exactly.

import torch
import torch.nn.functional as F

LAGS = [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 23, 24, 25, 35, 36, 37]
D_MODEL = 26


def _mean_scaler(context, observed_context):
    observed = observed_context.float()
    num = (context.abs() * observed).sum(dim=1, keepdim=True)
    den = observed.sum(dim=1, keepdim=True).clamp(min=1.0)
    scale = (num / den).clamp(min=1e-10)
    loc = torch.zeros_like(scale)
    return (context - loc) / scale, loc, scale


def _get_lagged_subsequences(sequence, lags, subseq_len, shift=0):
    full_len = sequence.shape[1]
    pieces = []
    for lag in lags:
        idx = lag - shift
        start = full_len - idx - subseq_len
        end   = -idx if idx > 0 else None
        pieces.append(sequence[:, start:end].unsqueeze(-1))
    return torch.cat(pieces, dim=-1)


def _build_static_feat(loc, scale, static_real_features, static_cat_features, cat_embedder_weight):
    squeezed_loc   = loc.squeeze(1)
    squeezed_scale = scale.squeeze(1)
    log_abs_loc = squeezed_loc.abs().log1p()
    log_scale   = squeezed_scale.log()
    static_feat = torch.cat([log_abs_loc, log_scale], dim=1)
    if static_real_features is not None:
        static_feat = torch.cat([static_real_features.float(), static_feat], dim=1)
    if static_cat_features is not None:
        emb = F.embedding(static_cat_features[:, 0].long(), cat_embedder_weight)
        static_feat = torch.cat([emb, static_feat], dim=1)
    return static_feat


def prepare_encoder_input(
    past_values, past_time_features, past_observed_mask,
    static_cat_features, static_real_features,
    cat_embedder_weight, value_proj_weight, pos_emb_weight,
    context_length=24,
):
    B = past_values.shape[0]
    device = past_values.device
    context          = past_values[:, -context_length:].float().unsqueeze(-1)
    observed_context = past_observed_mask[:, -context_length:].float().unsqueeze(-1)
    _, loc, scale    = _mean_scaler(context, observed_context)
    inputs_scaled    = ((past_values.float().unsqueeze(-1) - loc) / scale).squeeze(-1)
    lagged           = _get_lagged_subsequences(inputs_scaled, LAGS, context_length)
    reshaped_lagged  = lagged.reshape(B, context_length, -1)
    time_feat        = past_time_features[:, -context_length:].float()
    static_feat      = _build_static_feat(loc, scale, static_real_features, static_cat_features, cat_embedder_weight)
    expanded_static  = static_feat.unsqueeze(1).expand(-1, context_length, -1)
    features         = torch.cat([expanded_static, time_feat], dim=-1)
    transformer_inputs = torch.cat([reshaped_lagged, features], dim=-1)
    emb = transformer_inputs @ value_proj_weight.T
    positions = torch.arange(context_length, device=device)
    emb = emb + pos_emb_weight[positions].unsqueeze(0)
    return emb, loc, scale


def prepare_decoder_input(
    future_values, future_time_features, past_values,
    loc, scale,
    static_cat_features, static_real_features,
    cat_embedder_weight, value_proj_weight, pos_emb_weight,
    context_length=24,
    shift=0,
):
    B, pred_len = future_values.shape
    device = future_values.device
    full_seq         = torch.cat([past_values.float(), future_values.float()], dim=1)
    inputs_scaled    = ((full_seq.unsqueeze(-1) - loc) / scale).squeeze(-1)
    lagged           = _get_lagged_subsequences(inputs_scaled, LAGS, pred_len, shift=shift)
    reshaped_lagged  = lagged.reshape(B, pred_len, -1)
    time_feat        = future_time_features.float()
    static_feat      = _build_static_feat(loc, scale, static_real_features, static_cat_features, cat_embedder_weight)
    expanded_static  = static_feat.unsqueeze(1).expand(-1, pred_len, -1)
    features         = torch.cat([expanded_static, time_feat], dim=-1)
    transformer_inputs = torch.cat([reshaped_lagged, features], dim=-1)
    emb = transformer_inputs @ value_proj_weight.T
    positions = torch.arange(context_length, context_length + pred_len, device=device)
    emb = emb + pos_emb_weight[positions].unsqueeze(0)
    return emb

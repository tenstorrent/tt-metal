# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN-aligned PyTorch reference for Informer.
Reference from: https://github.com/zhouhaoyi/Informer2020

This file is a TTNN-aligned reference for weight mapping and fast correctness checks.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.demos.informer.tt.config import TILE_SIZE
from models.demos.informer.tt.ops import to_torch
from models.demos.informer.tt.state_io import export_torch_reference_state


def torch_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if x.dtype != weight.dtype:
        x = x.to(weight.dtype)
    if bias.dtype != weight.dtype:
        bias = bias.to(weight.dtype)
    return x @ weight.T + bias


def pad_to_multiple(x: torch.Tensor, multiple: int, *, dim: int = 1) -> Tuple[torch.Tensor, int]:
    length = x.shape[dim]
    pad = (multiple - (length % multiple)) % multiple
    if pad == 0:
        return x, length
    pad_shape = list(x.shape)
    pad_shape[dim] = pad
    pad_tensor = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad_tensor], dim=dim), length


def load_etth1_csv(path: Path, *, features: int) -> Tuple[object | None, torch.Tensor]:
    import pandas as pd

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Empty dataset.")
    time_col = df.columns[0]
    timestamps = pd.to_datetime(df[time_col], errors="coerce")
    if timestamps.notna().any():
        timestamps = timestamps.ffill().bfill()
        values = df.iloc[:, 1 : 1 + features]
    else:
        timestamps = None
        values = df.iloc[:, :features]
    values = values.astype("float32").to_numpy()
    if values.shape[1] < features:
        raise ValueError(f"Dataset has {values.shape[1]} features, expected {features}.")
    return timestamps, torch.tensor(values, dtype=torch.float32)


def build_calendar_time_features(timestamps: object | None, length: int, dim: int) -> torch.Tensor:
    if dim <= 0:
        return torch.zeros((length, 0), dtype=torch.float32)
    if timestamps is None:
        raise ValueError("Timestamps are required for calendar time features.")
    ts = timestamps.iloc[:length]
    month = (ts.dt.month - 1) / 11.0
    day = (ts.dt.day - 1) / 30.0
    weekday = ts.dt.weekday / 6.0
    hour = ts.dt.hour / 23.0
    minute = ts.dt.minute / 59.0
    features = [month, day, weekday, hour, minute]
    if dim > len(features):
        dayofyear = (ts.dt.dayofyear - 1) / 365.0
        features.append(dayofyear)
    stacked = torch.stack([torch.tensor(f.to_numpy(), dtype=torch.float32) for f in features[:dim]], dim=1)
    return stacked


def build_sinusoidal_time_features(length: int, dim: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    features = torch.zeros((length, dim), dtype=torch.float32)
    features[:, 0::2] = torch.sin(position * div_term)
    features[:, 1::2] = torch.cos(position * div_term)
    return features


def compute_normalization(values: torch.Tensor, train_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if train_len <= 0 or train_len > values.shape[0]:
        raise ValueError("Invalid train_len for normalization.")
    mean = values[:train_len].mean(dim=0)
    std = values[:train_len].std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def normalize_values(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (values - mean) / std


def denormalize_values(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return values * std + mean


def compute_metrics(pred: torch.Tensor, ref: torch.Tensor) -> Tuple[float, float, float]:
    diff = pred - ref
    mse = float((diff * diff).mean().item())
    mae = float(diff.abs().mean().item())
    x = pred.flatten()
    y = ref.flatten()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = torch.sqrt((vx * vx).sum()) * torch.sqrt((vy * vy).sum())
    corr = float((vx * vy).sum().item() / (denom.item() + 1e-8))
    corr = max(-1.0, min(1.0, corr))
    return mse, mae, corr


def make_causal_mask(length: int, mask_value: float, *, device) -> torch.Tensor:
    mask = torch.full((length, length), mask_value, dtype=torch.float32, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def make_padding_mask(valid_length: int, padded_length: int, mask_value: float, *, device) -> torch.Tensor:
    idx = torch.arange(padded_length, device=device)
    valid = (idx < valid_length).to(torch.float32)
    mask = (1.0 - valid) * mask_value
    return mask.view(1, 1, 1, padded_length)


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    batch, length, d_model = x.shape
    head_dim = d_model // n_heads
    x = x.view(batch, length, n_heads, head_dim)
    return x.transpose(1, 2).contiguous()


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, heads, length, head_dim = x.shape
    x = x.transpose(1, 2).contiguous()
    return x.view(batch, length, heads * head_dim)


def base_context_mean(v: torch.Tensor, valid_length: int) -> torch.Tensor:
    batch, heads, length, _ = v.shape
    mask = torch.arange(length, device=v.device) < valid_length
    mask = mask.view(1, 1, length, 1).expand(batch, heads, length, 1)
    mean = (v * mask.float()).sum(dim=2, keepdim=True) / max(1, valid_length)
    return mean.expand(batch, heads, length, v.shape[-1])


def base_context_cumsum(v: torch.Tensor) -> torch.Tensor:
    """Decoder base context uses cumulative sum over time, matching Informer ProbSparse."""
    return v.cumsum(dim=2, dtype=torch.float32).to(v.dtype)


def torch_mha_reference(
    attn,
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    prob_sparse: bool,
    factor: int,
    mask_value: float,
    dtype: torch.dtype = torch.float32,
    q_valid_len: int | None = None,
    k_valid_len: int | None = None,
) -> torch.Tensor:
    q = torch_linear(q_in, attn.q_weight_torch.to(dtype), attn.q_bias_torch.to(dtype))
    k = torch_linear(k_in, attn.k_weight_torch.to(dtype), attn.k_bias_torch.to(dtype))
    v = torch_linear(v_in, attn.v_weight_torch.to(dtype), attn.v_bias_torch.to(dtype))

    q, q_len = pad_to_multiple(q, TILE_SIZE, dim=1)
    k, k_len = pad_to_multiple(k, TILE_SIZE, dim=1)
    v, _ = pad_to_multiple(v, TILE_SIZE, dim=1)

    if q_valid_len is None:
        q_valid_len = q_len
    if k_valid_len is None:
        k_valid_len = k_len

    qh = split_heads(q, attn.n_heads)
    kh = split_heads(k, attn.n_heads)
    vh = split_heads(v, attn.n_heads)

    key_padding_mask = make_padding_mask(k_valid_len, k.shape[1], mask_value, device=q.device)
    attn_mask = key_padding_mask if mask is None else mask + key_padding_mask

    scores = qh @ kh.transpose(-2, -1)
    scores = scores / math.sqrt(attn.head_dim)
    scores_masked = scores
    if attn_mask is not None:
        scores_masked = scores + attn_mask.to(scores.dtype)
    attn_probs = torch.softmax(scores_masked, dim=-1)
    if attn_probs.dtype != vh.dtype:
        attn_probs = attn_probs.to(vh.dtype)
    ctx = attn_probs @ vh

    if prob_sparse:
        if attn_mask is not None and attn_mask.shape[1] == 1 and attn.n_heads > 1:
            attn_mask = attn_mask.expand(attn_mask.shape[0], attn.n_heads, attn_mask.shape[2], attn_mask.shape[3])
        if attn_mask is not None and attn_mask.shape[2] == 1 and q_len > 1:
            attn_mask = attn_mask.expand(attn_mask.shape[0], attn_mask.shape[1], q_len, attn_mask.shape[3])
        if attn_mask is not None and attn_mask.shape[0] == 1 and qh.shape[0] > 1:
            attn_mask = attn_mask.expand(qh.shape[0], attn_mask.shape[1], attn_mask.shape[2], attn_mask.shape[3])
        log_k = int(math.ceil(math.log1p(max(1, k_valid_len))))
        log_q = int(math.ceil(math.log1p(max(1, q_valid_len))))
        top_u = min(q_valid_len, max(1, int(factor * log_q)))
        if getattr(attn, "is_decoder", False):
            top_u = q_valid_len
        sample_k = min(k_valid_len, max(1, int(factor * q_valid_len * log_k)))
        step = max(1, k_valid_len // sample_k)
        idx = torch.arange(0, step * sample_k, step, device=q.device)
        kh_sample = kh[:, :, idx, :]
        sample_scores = qh @ kh_sample.transpose(-2, -1)
        sample_scores = sample_scores / math.sqrt(attn.head_dim)
        max_scores = sample_scores.max(dim=-1).values
        mean_scores = sample_scores.sum(dim=-1) / float(max(1, k_valid_len))
        sparsity = max_scores - mean_scores
        if q_valid_len < q_len:
            sparsity[:, :, q_valid_len:] = mask_value
        topk_idx = torch.topk(sparsity, k=top_u, dim=2, largest=True, sorted=False).indices
        qh_top = qh.gather(dim=2, index=topk_idx.unsqueeze(-1).expand(-1, -1, -1, qh.shape[-1]))
        scores_top = qh_top @ kh.transpose(-2, -1)
        scores_top = scores_top / math.sqrt(attn.head_dim)
        if attn_mask is not None:
            mask_top = attn_mask.gather(dim=2, index=topk_idx.unsqueeze(-1).expand(-1, -1, -1, attn_mask.shape[-1]))
            scores_top = scores_top + mask_top.to(scores_top.dtype)
        probs_top = torch.softmax(scores_top, dim=-1)
        if probs_top.dtype != vh.dtype:
            probs_top = probs_top.to(vh.dtype)
        ctx_top = probs_top @ vh
        if getattr(attn, "is_decoder", False):
            base_ctx = base_context_cumsum(vh)
        else:
            base_ctx = base_context_mean(vh, k_valid_len)
        if base_ctx.dtype != ctx_top.dtype:
            base_ctx = base_ctx.to(ctx_top.dtype)
        ctx = base_ctx.clone()
        ctx.scatter_(2, topk_idx.unsqueeze(-1).expand(-1, -1, -1, ctx_top.shape[-1]), ctx_top)

    ctx = merge_heads(ctx)
    ctx = ctx[:, :q_len, :]
    out = torch_linear(ctx, attn.o_weight_torch.to(dtype), attn.o_bias_torch.to(dtype))
    return out.float()


def informer_torch_forward(
    model,
    past_values: torch.Tensor,
    past_time_features: torch.Tensor,
    future_time_features: torch.Tensor,
    future_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """PyTorch reference implementation for Informer forward pass."""
    cfg = model.config
    compute_dtype = torch.float32
    pos = to_torch(model.embedding.positional_embedding.pe).float()
    enc_pos = pos[:, : cfg.seq_len, :]
    enc_embed = (
        torch_linear(
            past_values,
            model.embedding.value_embedding.weight_torch.to(compute_dtype),
            model.embedding.value_embedding.bias_torch.to(compute_dtype),
        )
        + torch_linear(
            past_time_features,
            model.embedding.temporal_embedding.weight_torch.to(compute_dtype),
            model.embedding.temporal_embedding.bias_torch.to(compute_dtype),
        )
        + enc_pos.to(compute_dtype)
    )

    enc = enc_embed
    enc_valid_len = enc.shape[1]
    enc_prob_sparse = cfg.attention_type == "prob"
    for i, layer in enumerate(model.encoder.layers):
        attn_out = torch_mha_reference(
            layer.attn,
            enc,
            enc,
            enc,
            None,
            prob_sparse=enc_prob_sparse,
            factor=cfg.factor,
            mask_value=cfg.attn_mask_value,
            dtype=compute_dtype,
            q_valid_len=enc_valid_len,
            k_valid_len=enc_valid_len,
        )
        enc = torch.nn.functional.layer_norm(
            (enc + attn_out).to(compute_dtype),
            (cfg.d_model,),
            weight=layer.norm1.weight_torch.to(compute_dtype),
            bias=layer.norm1.bias_torch.to(compute_dtype),
            eps=layer.norm1.eps,
        )
        ff = torch_linear(
            enc,
            layer.ffn.w1_torch.to(compute_dtype),
            layer.ffn.b1_torch.to(compute_dtype),
        )
        ff = torch.nn.functional.gelu(ff)
        ff = torch_linear(
            ff,
            layer.ffn.w2_torch.to(compute_dtype),
            layer.ffn.b2_torch.to(compute_dtype),
        )
        enc = torch.nn.functional.layer_norm(
            (enc + ff).to(compute_dtype),
            (cfg.d_model,),
            weight=layer.norm2.weight_torch.to(compute_dtype),
            bias=layer.norm2.bias_torch.to(compute_dtype),
            eps=layer.norm2.eps,
        )
        if cfg.distil.enabled and i < len(model.encoder.layers) - 1:
            enc = torch.nn.functional.max_pool1d(
                enc.permute(0, 2, 1),
                kernel_size=cfg.distil.kernel_size,
                stride=cfg.distil.stride,
                padding=cfg.distil.padding,
            ).permute(0, 2, 1)
            enc_valid_len = max(
                1,
                (enc_valid_len + 2 * cfg.distil.padding - cfg.distil.kernel_size) // cfg.distil.stride + 1,
            )
            enc = torch.nn.functional.layer_norm(
                enc.to(compute_dtype),
                (cfg.d_model,),
                weight=model.encoder.distil_norm.weight_torch.to(compute_dtype),
                bias=model.encoder.distil_norm.bias_torch.to(compute_dtype),
                eps=model.encoder.distil_norm.eps,
            )

    label_len = cfg.label_len
    dec_known = past_values[:, -label_len:, :]
    if future_values is None:
        future_pad = torch.zeros(past_values.shape[0], cfg.pred_len, cfg.dec_in, dtype=torch.float32)
    else:
        future_pad = future_values[:, : cfg.pred_len, :]
    dec_values = torch.cat([dec_known, future_pad], dim=1)
    dec_time = torch.cat([past_time_features[:, -label_len:, :], future_time_features[:, : cfg.pred_len, :]], dim=1)

    dec_pos = pos[:, : dec_values.shape[1], :]
    dec_embed = (
        torch_linear(
            dec_values,
            model.embedding.value_embedding.weight_torch.to(compute_dtype),
            model.embedding.value_embedding.bias_torch.to(compute_dtype),
        )
        + torch_linear(
            dec_time,
            model.embedding.temporal_embedding.weight_torch.to(compute_dtype),
            model.embedding.temporal_embedding.bias_torch.to(compute_dtype),
        )
        + dec_pos.to(compute_dtype)
    )

    dec = dec_embed
    dec_valid_len = dec.shape[1]
    dec_len = dec.shape[1]
    pad_len = int(math.ceil(dec_len / TILE_SIZE)) * TILE_SIZE
    causal_mask = make_causal_mask(pad_len, cfg.attn_mask_value, device=dec.device)
    dec_prob_sparse = False
    for layer in model.decoder.layers:
        attn1 = torch_mha_reference(
            layer.self_attn,
            dec,
            dec,
            dec,
            causal_mask,
            prob_sparse=dec_prob_sparse,
            factor=cfg.factor,
            mask_value=cfg.attn_mask_value,
            dtype=compute_dtype,
            q_valid_len=dec_valid_len,
            k_valid_len=dec_valid_len,
        )
        dec = torch.nn.functional.layer_norm(
            (dec + attn1).to(compute_dtype),
            (cfg.d_model,),
            weight=layer.norm1.weight_torch.to(compute_dtype),
            bias=layer.norm1.bias_torch.to(compute_dtype),
            eps=layer.norm1.eps,
        )
        attn2 = torch_mha_reference(
            layer.cross_attn,
            dec,
            enc,
            enc,
            None,
            prob_sparse=False,
            factor=cfg.factor,
            mask_value=cfg.attn_mask_value,
            dtype=compute_dtype,
            q_valid_len=dec_valid_len,
            k_valid_len=enc_valid_len,
        )
        dec = torch.nn.functional.layer_norm(
            (dec + attn2).to(compute_dtype),
            (cfg.d_model,),
            weight=layer.norm2.weight_torch.to(compute_dtype),
            bias=layer.norm2.bias_torch.to(compute_dtype),
            eps=layer.norm2.eps,
        )
        ff = torch_linear(
            dec,
            layer.ffn.w1_torch.to(compute_dtype),
            layer.ffn.b1_torch.to(compute_dtype),
        )
        ff = torch.nn.functional.gelu(ff)
        ff = torch_linear(
            ff,
            layer.ffn.w2_torch.to(compute_dtype),
            layer.ffn.b2_torch.to(compute_dtype),
        )
        dec = torch.nn.functional.layer_norm(
            (dec + ff).to(compute_dtype),
            (cfg.d_model,),
            weight=layer.norm3.weight_torch.to(compute_dtype),
            bias=layer.norm3.bias_torch.to(compute_dtype),
            eps=layer.norm3.eps,
        )

    pred = torch_linear(
        dec[:, -cfg.pred_len :, :],
        model.proj_w_torch.to(compute_dtype),
        model.proj_b_torch.to(compute_dtype),
    ).float()
    return pred


class TorchInformerEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.value_embedding = nn.Linear(cfg.enc_in, cfg.d_model, bias=True)
        self.temporal_embedding = nn.Linear(cfg.time_feature_dim, cfg.d_model, bias=True)
        max_len = cfg.seq_len + cfg.pred_len + cfg.label_len
        self.register_buffer("positional_embedding", build_positional_encoding(max_len, cfg.d_model))
        self.dropout = cfg.dropout

    def forward(self, values: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        pos = self.positional_embedding[:, : values.shape[1], :]
        out = self.value_embedding(values) + self.temporal_embedding(time_features) + pos
        return F.dropout(out, p=self.dropout, training=self.training)


class TorchMultiHeadAttention(nn.Module):
    def __init__(self, cfg, *, prob_sparse: bool, is_decoder: bool = False):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout
        self.prob_sparse = prob_sparse
        self.is_decoder = is_decoder
        self.factor = cfg.factor
        self.mask_value = cfg.attn_mask_value
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        q_valid_len: int | None = None,
        k_valid_len: int | None = None,
    ) -> torch.Tensor:
        batch, q_len, _ = query.shape
        _, k_len, _ = key.shape
        if q_valid_len is None:
            q_valid_len = q_len
        if k_valid_len is None:
            k_valid_len = k_len

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q, q_len_pad = pad_to_multiple(q, TILE_SIZE, dim=1)
        k, _ = pad_to_multiple(k, TILE_SIZE, dim=1)
        v, _ = pad_to_multiple(v, TILE_SIZE, dim=1)

        qh = split_heads(q, self.n_heads)
        kh = split_heads(k, self.n_heads)
        vh = split_heads(v, self.n_heads)

        key_padding_mask = make_padding_mask(k_valid_len, k.shape[1], self.mask_value, device=q.device)
        attn_mask = key_padding_mask if mask is None else mask + key_padding_mask
        if not self.prob_sparse:
            scores = qh @ kh.transpose(-2, -1)
            scores = scores / math.sqrt(self.head_dim)
            if attn_mask is not None:
                scores = scores + attn_mask.to(scores.dtype)
            attn_probs = torch.softmax(scores, dim=-1)
            if attn_probs.dtype != vh.dtype:
                attn_probs = attn_probs.to(vh.dtype)
            context = attn_probs @ vh
        else:
            if attn_mask is not None and attn_mask.shape[1] == 1 and self.n_heads > 1:
                attn_mask = attn_mask.expand(attn_mask.shape[0], self.n_heads, attn_mask.shape[2], attn_mask.shape[3])
            if attn_mask is not None and attn_mask.shape[2] == 1 and q_len_pad > 1:
                attn_mask = attn_mask.expand(attn_mask.shape[0], attn_mask.shape[1], q_len_pad, attn_mask.shape[3])
            if attn_mask is not None and attn_mask.shape[0] == 1 and qh.shape[0] > 1:
                attn_mask = attn_mask.expand(qh.shape[0], attn_mask.shape[1], attn_mask.shape[2], attn_mask.shape[3])

            log_k = int(math.ceil(math.log1p(max(1, k_valid_len))))
            log_q = int(math.ceil(math.log1p(max(1, q_valid_len))))
            top_u = min(q_valid_len, max(1, int(self.factor * log_q)))
            sample_k = min(k_valid_len, max(1, int(self.factor * q_valid_len * log_k)))
            step = max(1, k_valid_len // sample_k)
            idx = torch.arange(0, step * sample_k, step, device=q.device)
            kh_sample = kh[:, :, idx, :]
            sample_scores = qh @ kh_sample.transpose(-2, -1)
            sample_scores = sample_scores / math.sqrt(self.head_dim)
            max_scores = sample_scores.max(dim=-1).values
            mean_scores = sample_scores.sum(dim=-1) / float(max(1, k_valid_len))
            sparsity = max_scores - mean_scores
            if q_valid_len < q_len_pad:
                sparsity[:, :, q_valid_len:] = self.mask_value

            topk_idx = torch.topk(sparsity, k=top_u, dim=2, largest=True, sorted=False).indices

            qh_top = qh.gather(dim=2, index=topk_idx.unsqueeze(-1).expand(-1, -1, -1, qh.shape[-1]))
            scores_top = qh_top @ kh.transpose(-2, -1)
            scores_top = scores_top / math.sqrt(self.head_dim)
            if attn_mask is not None:
                mask_top = attn_mask.gather(dim=2, index=topk_idx.unsqueeze(-1).expand(-1, -1, -1, attn_mask.shape[-1]))
                scores_top = scores_top + mask_top.to(scores_top.dtype)
            probs_top = torch.softmax(scores_top, dim=-1)
            if probs_top.dtype != vh.dtype:
                probs_top = probs_top.to(vh.dtype)
            ctx_top = probs_top @ vh

            if self.is_decoder:
                base_ctx = base_context_cumsum(vh)
            else:
                base_ctx = base_context_mean(vh, k_valid_len)
            if base_ctx.dtype != ctx_top.dtype:
                base_ctx = base_ctx.to(ctx_top.dtype)
            context = base_ctx.clone()
            context.scatter_(2, topk_idx.unsqueeze(-1).expand(-1, -1, -1, ctx_top.shape[-1]), ctx_top)

        context = merge_heads(context)
        context = context[:, :q_len, :]
        out = self.o_proj(context)
        return F.dropout(out, p=self.dropout, training=self.training)


class TorchFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=True)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=True)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return F.dropout(x, p=self.dropout, training=self.training)


class TorchEncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = TorchMultiHeadAttention(cfg, prob_sparse=cfg.attention_type == "prob", is_decoder=False)
        self.ffn = TorchFeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None, valid_len: int) -> torch.Tensor:
        attn_out = self.attn(x, x, x, mask, q_valid_len=valid_len, k_valid_len=valid_len)
        x = self.norm1(x + attn_out)
        ff = self.ffn(x)
        x = self.norm2(x + ff)
        return x


class TorchEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([TorchEncoderLayer(cfg) for _ in range(cfg.e_layers)])
        self.distil = cfg.distil
        self.distil_norm = nn.LayerNorm(cfg.d_model) if self.distil.enabled else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, int]:
        valid_len = x.shape[1]
        for i, layer in enumerate(self.layers):
            x = layer(x, mask, valid_len)
            if self.distil.enabled and i < len(self.layers) - 1:
                x = F.max_pool1d(
                    x.permute(0, 2, 1),
                    kernel_size=self.distil.kernel_size,
                    stride=self.distil.stride,
                    padding=self.distil.padding,
                ).permute(0, 2, 1)
                valid_len = max(
                    1,
                    (valid_len + 2 * self.distil.padding - self.distil.kernel_size) // self.distil.stride + 1,
                )
                x = self.distil_norm(x)
        return x, valid_len


class TorchDecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = TorchMultiHeadAttention(cfg, prob_sparse=False, is_decoder=True)
        self.cross_attn = TorchMultiHeadAttention(cfg, prob_sparse=False, is_decoder=False)
        self.ffn = TorchFeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.norm3 = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        self_mask: torch.Tensor | None,
        cross_mask: torch.Tensor | None,
        enc_valid_len: int,
    ) -> torch.Tensor:
        attn1 = self.self_attn(x, x, x, self_mask, q_valid_len=x.shape[1], k_valid_len=x.shape[1])
        x = self.norm1(x + attn1)
        attn2 = self.cross_attn(x, enc_out, enc_out, cross_mask, q_valid_len=x.shape[1], k_valid_len=enc_valid_len)
        x = self.norm2(x + attn2)
        ff = self.ffn(x)
        x = self.norm3(x + ff)
        return x


class TorchDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([TorchDecoderLayer(cfg) for _ in range(cfg.d_layers)])

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        self_mask: torch.Tensor | None,
        cross_mask: torch.Tensor | None,
        enc_valid_len: int,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, enc_out, self_mask, cross_mask, enc_valid_len)
        return x


class TorchInformerModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.embedding = TorchInformerEmbedding(cfg)
        self.encoder = TorchEncoder(cfg)
        self.decoder = TorchDecoder(cfg)
        self.proj = nn.Linear(cfg.d_model, cfg.c_out, bias=True)

    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        future_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg = self.config
        enc_embed = self.embedding(past_values, past_time_features)
        enc_out, enc_valid_len = self.encoder(enc_embed, None)

        label_len = cfg.label_len
        dec_known = past_values[:, -label_len:, :]
        if future_values is None:
            future_pad = torch.zeros(
                (past_values.shape[0], cfg.pred_len, cfg.dec_in),
                dtype=past_values.dtype,
                device=past_values.device,
            )
        else:
            future_pad = future_values[:, : cfg.pred_len, :]
        dec_values = torch.cat([dec_known, future_pad], dim=1)
        dec_time = torch.cat([past_time_features[:, -label_len:, :], future_time_features[:, : cfg.pred_len, :]], dim=1)
        dec_embed = self.embedding(dec_values, dec_time)

        dec_len = dec_embed.shape[1]
        pad_len = int(math.ceil(dec_len / TILE_SIZE)) * TILE_SIZE
        self_mask = make_causal_mask(pad_len, cfg.attn_mask_value, device=dec_embed.device)
        dec_out = self.decoder(dec_embed, enc_out, self_mask, None, enc_valid_len)
        return self.proj(dec_out[:, -cfg.pred_len :, :])


def ttnn_state_dict(model: TorchInformerModel) -> dict[str, torch.Tensor]:
    return export_torch_reference_state(model)


def build_positional_encoding(length: int, d_model: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32)[:, None]
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros((length, d_model), dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

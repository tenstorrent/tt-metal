# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.demos.informer.tt import TILE_SIZE, to_torch


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


@dataclass
class SplitConfig:
    train_len: int
    val_len: int
    test_len: int


def default_etth1_splits() -> SplitConfig:
    return SplitConfig(
        train_len=12 * 30 * 24,
        val_len=4 * 30 * 24,
        test_len=4 * 30 * 24,
    )


def load_etth1_csv(path: Path, *, features: int) -> Tuple[object | None, torch.Tensor]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for CSV loading. Install pandas or provide preloaded tensors.") from exc

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Empty dataset.")
    time_col = df.columns[0]
    timestamps = None
    try:
        timestamps = pd.to_datetime(df[time_col])
        values = df.iloc[:, 1 : 1 + features]
    except Exception:
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


def iter_windows(
    total_length: int,
    *,
    seq_len: int,
    pred_len: int,
    stride: int,
    max_windows: int | None = None,
) -> Iterable[int]:
    total_window = seq_len + pred_len
    count = 0
    for start in range(0, total_length - total_window + 1, stride):
        yield start
        count += 1
        if max_windows is not None and count >= max_windows:
            break


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


def prob_sparse_mask(q: torch.Tensor, top_u: int, valid_length: int, mask_value: float) -> torch.Tensor:
    batch, heads, length, _ = q.shape
    q_norm = torch.sum(q * q, dim=-1)
    if valid_length < length:
        q_norm[:, :, valid_length:] = mask_value
    topk_idx = torch.topk(q_norm, k=top_u, dim=-1, largest=True, sorted=False).indices
    mask = torch.zeros((batch, heads, length), device=q.device, dtype=torch.float32)
    mask.scatter_(2, topk_idx, 1.0)
    return mask.unsqueeze(-1)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    numeric_stable: bool,
) -> torch.Tensor:
    head_dim = q.shape[-1]
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(head_dim)
    scores = scores.to(torch.bfloat16)
    if mask is not None:
        scores = scores + mask.to(scores.dtype)
    if numeric_stable:
        probs = torch.softmax(scores, dim=-1)
    else:
        exp_scores = torch.exp(scores)
        probs = exp_scores / exp_scores.sum(dim=-1, keepdim=True)
    if probs.dtype != v.dtype:
        probs = probs.to(v.dtype)
    return probs @ v


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
) -> torch.Tensor:
    q = torch_linear(q_in, attn.q_weight_torch.to(torch.bfloat16), attn.q_bias_torch.to(torch.bfloat16))
    k = torch_linear(k_in, attn.k_weight_torch.to(torch.bfloat16), attn.k_bias_torch.to(torch.bfloat16))
    v = torch_linear(v_in, attn.v_weight_torch.to(torch.bfloat16), attn.v_bias_torch.to(torch.bfloat16))

    q, q_len = pad_to_multiple(q, TILE_SIZE, dim=1)
    k, k_len = pad_to_multiple(k, TILE_SIZE, dim=1)
    v, _ = pad_to_multiple(v, TILE_SIZE, dim=1)

    qh = split_heads(q, attn.n_heads)
    kh = split_heads(k, attn.n_heads)
    vh = split_heads(v, attn.n_heads)

    key_padding_mask = make_padding_mask(k_len, k.shape[1], mask_value, device=q.device)
    attn_mask = key_padding_mask if mask is None else mask + key_padding_mask
    use_stable = attn_mask is not None and attn_mask.shape[2] > 1
    ctx = scaled_dot_product_attention(qh, kh, vh, attn_mask, numeric_stable=use_stable)

    if prob_sparse:
        u = max(1, int(factor * math.log(max(2, q_len))))
        ps_mask = prob_sparse_mask(qh, u, q_len, mask_value)
        ps_mask = ps_mask.repeat(1, 1, 1, attn.d_model // attn.n_heads)
        ctx = ctx * ps_mask

    ctx = merge_heads(ctx)
    ctx = ctx[:, :q_len, :]
    out = torch_linear(ctx, attn.o_weight_torch.to(torch.bfloat16), attn.o_bias_torch.to(torch.bfloat16))
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
    pos = to_torch(model.embedding.positional_embedding.pe).float()
    enc_pos = pos[:, : cfg.seq_len, :]
    enc_embed = (
        torch_linear(
            past_values,
            model.embedding.value_embedding.weight_torch.to(torch.bfloat16),
            model.embedding.value_embedding.bias_torch.to(torch.bfloat16),
        )
        + torch_linear(
            past_time_features,
            model.embedding.temporal_embedding.weight_torch.to(torch.bfloat16),
            model.embedding.temporal_embedding.bias_torch.to(torch.bfloat16),
        )
        + enc_pos.to(torch.bfloat16)
    )

    enc = enc_embed
    for i, layer in enumerate(model.encoder.layers):
        attn_out = torch_mha_reference(
            layer.attn,
            enc,
            enc,
            enc,
            None,
            prob_sparse=True,
            factor=cfg.factor,
            mask_value=cfg.attn_mask_value,
        )
        enc = torch.nn.functional.layer_norm(
            (enc + attn_out).to(torch.bfloat16),
            (cfg.d_model,),
            weight=layer.norm1.weight_torch.to(torch.bfloat16),
            bias=layer.norm1.bias_torch.to(torch.bfloat16),
            eps=layer.norm1.eps,
        )
        ff = torch_linear(
            enc,
            layer.ffn.w1_torch.to(torch.bfloat16),
            layer.ffn.b1_torch.to(torch.bfloat16),
        )
        ff = torch.nn.functional.gelu(ff)
        ff = torch_linear(
            ff,
            layer.ffn.w2_torch.to(torch.bfloat16),
            layer.ffn.b2_torch.to(torch.bfloat16),
        )
        enc = torch.nn.functional.layer_norm(
            (enc + ff).to(torch.bfloat16),
            (cfg.d_model,),
            weight=layer.norm2.weight_torch.to(torch.bfloat16),
            bias=layer.norm2.bias_torch.to(torch.bfloat16),
            eps=layer.norm2.eps,
        )
        if cfg.distil.enabled and i < len(model.encoder.layers) - 1:
            enc = torch.nn.functional.max_pool1d(
                enc.permute(0, 2, 1),
                kernel_size=cfg.distil.kernel_size,
                stride=cfg.distil.stride,
                padding=cfg.distil.padding,
            ).permute(0, 2, 1)
            enc, _ = pad_to_multiple(enc, TILE_SIZE, dim=1)
            enc = torch.nn.functional.layer_norm(
                enc.to(torch.bfloat16),
                (cfg.d_model,),
                weight=model.encoder.distil_norm.weight_torch.to(torch.bfloat16),
                bias=model.encoder.distil_norm.bias_torch.to(torch.bfloat16),
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
            model.embedding.value_embedding.weight_torch.to(torch.bfloat16),
            model.embedding.value_embedding.bias_torch.to(torch.bfloat16),
        )
        + torch_linear(
            dec_time,
            model.embedding.temporal_embedding.weight_torch.to(torch.bfloat16),
            model.embedding.temporal_embedding.bias_torch.to(torch.bfloat16),
        )
        + dec_pos.to(torch.bfloat16)
    )

    dec = dec_embed
    dec_len = dec.shape[1]
    pad_len = int(math.ceil(dec_len / TILE_SIZE)) * TILE_SIZE
    causal_mask = make_causal_mask(pad_len, cfg.attn_mask_value, device=dec.device)
    for layer in model.decoder.layers:
        attn1 = torch_mha_reference(
            layer.self_attn,
            dec,
            dec,
            dec,
            causal_mask,
            prob_sparse=False,
            factor=cfg.factor,
            mask_value=cfg.attn_mask_value,
        )
        dec = torch.nn.functional.layer_norm(
            (dec + attn1).to(torch.bfloat16),
            (cfg.d_model,),
            weight=layer.norm1.weight_torch.to(torch.bfloat16),
            bias=layer.norm1.bias_torch.to(torch.bfloat16),
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
        )
        dec = torch.nn.functional.layer_norm(
            (dec + attn2).to(torch.bfloat16),
            (cfg.d_model,),
            weight=layer.norm2.weight_torch.to(torch.bfloat16),
            bias=layer.norm2.bias_torch.to(torch.bfloat16),
            eps=layer.norm2.eps,
        )
        ff = torch_linear(
            dec,
            layer.ffn.w1_torch.to(torch.bfloat16),
            layer.ffn.b1_torch.to(torch.bfloat16),
        )
        ff = torch.nn.functional.gelu(ff)
        ff = torch_linear(
            ff,
            layer.ffn.w2_torch.to(torch.bfloat16),
            layer.ffn.b2_torch.to(torch.bfloat16),
        )
        dec = torch.nn.functional.layer_norm(
            (dec + ff).to(torch.bfloat16),
            (cfg.d_model,),
            weight=layer.norm3.weight_torch.to(torch.bfloat16),
            bias=layer.norm3.bias_torch.to(torch.bfloat16),
            eps=layer.norm3.eps,
        )

    pred = torch_linear(
        dec[:, -cfg.pred_len :, :],
        model.proj_w_torch.to(torch.bfloat16),
        model.proj_b_torch.to(torch.bfloat16),
    ).float()
    return pred


class TorchInformerEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.value_embedding = nn.Linear(cfg.enc_in, cfg.d_model, bias=True)
        self.temporal_embedding = nn.Linear(cfg.time_feature_dim, cfg.d_model, bias=True)
        max_len = cfg.seq_len + cfg.pred_len + cfg.label_len
        self.register_buffer("positional_embedding", _build_positional_encoding(max_len, cfg.d_model))
        self.dropout = cfg.dropout

    def forward(self, values: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        pos = self.positional_embedding[:, : values.shape[1], :]
        out = self.value_embedding(values) + self.temporal_embedding(time_features) + pos
        return F.dropout(out, p=self.dropout, training=self.training)


class TorchMultiHeadAttention(nn.Module):
    def __init__(self, cfg, *, prob_sparse: bool):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout
        self.prob_sparse = prob_sparse
        self.factor = cfg.factor
        self.mask_value = cfg.attn_mask_value
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return split_heads(x, self.n_heads)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        return merge_heads(x)

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
        k, k_len_pad = pad_to_multiple(k, TILE_SIZE, dim=1)
        v, _ = pad_to_multiple(v, TILE_SIZE, dim=1)

        qh = self._split_heads(q)
        kh = self._split_heads(k)
        vh = self._split_heads(v)

        key_padding_mask = make_padding_mask(k_valid_len, k.shape[1], self.mask_value, device=q.device)
        attn_mask = key_padding_mask if mask is None else mask + key_padding_mask
        use_stable = attn_mask is not None and attn_mask.shape[2] > 1
        context = scaled_dot_product_attention(qh, kh, vh, attn_mask, numeric_stable=use_stable)

        if self.prob_sparse:
            u = max(1, int(self.factor * math.log(max(2, q_valid_len))))
            ps_mask = prob_sparse_mask(qh, u, q_len_pad, self.mask_value)
            ps_mask = ps_mask.repeat(1, 1, 1, self.d_model // self.n_heads)
            context = context * ps_mask

        context = self._merge_heads(context)
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
        self.attn = TorchMultiHeadAttention(cfg, prob_sparse=True)
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
                x, _ = pad_to_multiple(x, TILE_SIZE, dim=1)
                x = self.distil_norm(x)
        return x, valid_len


class TorchDecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = TorchMultiHeadAttention(cfg, prob_sparse=False)
        self.cross_attn = TorchMultiHeadAttention(cfg, prob_sparse=False)
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

    def state_dict_ttnn(self) -> dict[str, torch.Tensor]:
        cfg = self.config
        state: dict[str, torch.Tensor] = {}
        state["embedding.value.weight"] = self.embedding.value_embedding.weight.detach().cpu()
        state["embedding.value.bias"] = self.embedding.value_embedding.bias.detach().cpu()
        state["embedding.temporal.weight"] = self.embedding.temporal_embedding.weight.detach().cpu()
        state["embedding.temporal.bias"] = self.embedding.temporal_embedding.bias.detach().cpu()

        for i, layer in enumerate(self.encoder.layers):
            prefix = f"encoder.layers.{i}"
            attn = layer.attn
            state[f"{prefix}.attn.q_weight"] = attn.q_proj.weight.detach().cpu()
            state[f"{prefix}.attn.q_bias"] = attn.q_proj.bias.detach().cpu()
            state[f"{prefix}.attn.k_weight"] = attn.k_proj.weight.detach().cpu()
            state[f"{prefix}.attn.k_bias"] = attn.k_proj.bias.detach().cpu()
            state[f"{prefix}.attn.v_weight"] = attn.v_proj.weight.detach().cpu()
            state[f"{prefix}.attn.v_bias"] = attn.v_proj.bias.detach().cpu()
            state[f"{prefix}.attn.o_weight"] = attn.o_proj.weight.detach().cpu()
            state[f"{prefix}.attn.o_bias"] = attn.o_proj.bias.detach().cpu()

            state[f"{prefix}.ffn.w1"] = layer.ffn.fc1.weight.detach().cpu()
            state[f"{prefix}.ffn.b1"] = layer.ffn.fc1.bias.detach().cpu()
            state[f"{prefix}.ffn.w2"] = layer.ffn.fc2.weight.detach().cpu()
            state[f"{prefix}.ffn.b2"] = layer.ffn.fc2.bias.detach().cpu()

            state[f"{prefix}.norm1.weight"] = layer.norm1.weight.detach().cpu()
            state[f"{prefix}.norm1.bias"] = layer.norm1.bias.detach().cpu()
            state[f"{prefix}.norm2.weight"] = layer.norm2.weight.detach().cpu()
            state[f"{prefix}.norm2.bias"] = layer.norm2.bias.detach().cpu()

        if self.encoder.distil_norm is not None:
            state["encoder.distil_norm.weight"] = self.encoder.distil_norm.weight.detach().cpu()
            state["encoder.distil_norm.bias"] = self.encoder.distil_norm.bias.detach().cpu()

        for i, layer in enumerate(self.decoder.layers):
            prefix = f"decoder.layers.{i}"
            attn = layer.self_attn
            state[f"{prefix}.self_attn.q_weight"] = attn.q_proj.weight.detach().cpu()
            state[f"{prefix}.self_attn.q_bias"] = attn.q_proj.bias.detach().cpu()
            state[f"{prefix}.self_attn.k_weight"] = attn.k_proj.weight.detach().cpu()
            state[f"{prefix}.self_attn.k_bias"] = attn.k_proj.bias.detach().cpu()
            state[f"{prefix}.self_attn.v_weight"] = attn.v_proj.weight.detach().cpu()
            state[f"{prefix}.self_attn.v_bias"] = attn.v_proj.bias.detach().cpu()
            state[f"{prefix}.self_attn.o_weight"] = attn.o_proj.weight.detach().cpu()
            state[f"{prefix}.self_attn.o_bias"] = attn.o_proj.bias.detach().cpu()

            attn = layer.cross_attn
            state[f"{prefix}.cross_attn.q_weight"] = attn.q_proj.weight.detach().cpu()
            state[f"{prefix}.cross_attn.q_bias"] = attn.q_proj.bias.detach().cpu()
            state[f"{prefix}.cross_attn.k_weight"] = attn.k_proj.weight.detach().cpu()
            state[f"{prefix}.cross_attn.k_bias"] = attn.k_proj.bias.detach().cpu()
            state[f"{prefix}.cross_attn.v_weight"] = attn.v_proj.weight.detach().cpu()
            state[f"{prefix}.cross_attn.v_bias"] = attn.v_proj.bias.detach().cpu()
            state[f"{prefix}.cross_attn.o_weight"] = attn.o_proj.weight.detach().cpu()
            state[f"{prefix}.cross_attn.o_bias"] = attn.o_proj.bias.detach().cpu()

            state[f"{prefix}.ffn.w1"] = layer.ffn.fc1.weight.detach().cpu()
            state[f"{prefix}.ffn.b1"] = layer.ffn.fc1.bias.detach().cpu()
            state[f"{prefix}.ffn.w2"] = layer.ffn.fc2.weight.detach().cpu()
            state[f"{prefix}.ffn.b2"] = layer.ffn.fc2.bias.detach().cpu()

            state[f"{prefix}.norm1.weight"] = layer.norm1.weight.detach().cpu()
            state[f"{prefix}.norm1.bias"] = layer.norm1.bias.detach().cpu()
            state[f"{prefix}.norm2.weight"] = layer.norm2.weight.detach().cpu()
            state[f"{prefix}.norm2.bias"] = layer.norm2.bias.detach().cpu()
            state[f"{prefix}.norm3.weight"] = layer.norm3.weight.detach().cpu()
            state[f"{prefix}.norm3.bias"] = layer.norm3.bias.detach().cpu()

        state["projection.weight"] = self.proj.weight.detach().cpu()
        state["projection.bias"] = self.proj.bias.detach().cpu()
        return state


def _build_positional_encoding(length: int, d_model: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32)[:, None]
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros((length, d_model), dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

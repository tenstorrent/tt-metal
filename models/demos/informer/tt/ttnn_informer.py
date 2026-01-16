# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

import ttnn

TILE_SIZE = 32


@dataclass
class DistilConfig:
    enabled: bool = True
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1


@dataclass
class InformerConfig:
    # Input / output
    enc_in: int = 7
    dec_in: int = 7
    c_out: int = 7
    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 24

    # Model dims
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048

    # Depth
    e_layers: int = 2
    d_layers: int = 1

    # Attention / sparsity
    factor: int = 5
    dropout: float = 0.0

    # Positional / temporal embedding
    embed_dim: Optional[int] = None
    time_feature_dim: int = 4

    # TTNN runtime
    device_id: int = 0
    dtype: str = "bfloat16"
    attn_mask_value: float = -1e4

    # Distilling
    distil: DistilConfig = field(default_factory=DistilConfig)

    def __post_init__(self):
        if self.embed_dim is None:
            self.embed_dim = self.d_model
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads for head splits.")
        head_dim = self.d_model // self.n_heads
        if head_dim % TILE_SIZE != 0:
            raise ValueError("head_dim must be a multiple of 32 for TTNN tile matmul.")
        if self.d_model % TILE_SIZE != 0:
            raise ValueError("d_model must be a multiple of 32 for TTNN tile matmul.")


def get_ttnn_dtype(dtype: str) -> ttnn.DataType:
    if dtype == "bfloat16":
        return ttnn.bfloat16
    if dtype == "float32":
        return ttnn.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def to_torch(x: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(x)


def make_causal_mask(
    length: int, *, batch: int, heads: int, device, dtype: ttnn.DataType, mask_value: float
) -> ttnn.Tensor:
    mask = torch.full((length, length), mask_value, dtype=torch.float32)
    mask = torch.triu(mask, diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = ttnn.from_torch(mask, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    if batch > 1 or heads > 1:
        mask = ttnn.repeat(mask, (batch, heads, 1, 1))
    return mask


def make_padding_mask(
    *,
    valid_length: int,
    key_length: int,
    batch: int,
    heads: int,
    device,
    dtype: ttnn.DataType,
    mask_value: float,
) -> ttnn.Tensor:
    idx_range = ttnn.arange(end=key_length, device=device, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT)
    idx_range = ttnn.reshape(idx_range, (1, 1, 1, key_length))
    valid = ttnn.lt(idx_range, valid_length, dtype=ttnn.bfloat16)
    mask = ttnn.mul(valid, -1.0)
    mask = mask + 1.0
    mask = mask * mask_value
    mask = ttnn.repeat(mask, (batch, heads, 1, 1))
    mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)
    return mask


def sinusoidal_position_encoding(length: int, d_model: int, *, device, dtype: ttnn.DataType) -> ttnn.Tensor:
    position = torch.arange(length, dtype=torch.float32)[:, None]
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros((length, d_model), dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return ttnn.from_torch(pe, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)


def pad_to_multiple(x: ttnn.Tensor, *, dim: int, multiple: int, value: float = 0.0) -> Tuple[ttnn.Tensor, int]:
    length = x.shape[dim]
    pad = (multiple - (length % multiple)) % multiple
    if pad == 0:
        return x, length
    padding = [(0, 0)] * len(x.shape)
    padding[dim] = (0, pad)
    x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.pad(x, padding, value=value, memory_config=ttnn.DRAM_MEMORY_CONFIG), length


def slice_to_length(x: ttnn.Tensor, *, dim: int, length: int) -> ttnn.Tensor:
    if x.shape[dim] == length:
        return x
    slice_start = [0] * len(x.shape)
    slice_end = list(x.shape)
    slice_end[dim] = length
    return ttnn.slice(x, slice_start, slice_end)


def linear(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    bias: Optional[ttnn.Tensor] = None,
    *,
    dtype: Optional[ttnn.DataType] = None,
) -> ttnn.Tensor:
    return ttnn.linear(x, weight, bias=bias, transpose_b=True, dtype=dtype)


def max_pool1d(
    x: ttnn.Tensor,
    *,
    kernel: int,
    stride: int,
    padding: int,
    dtype: ttnn.DataType,
) -> ttnn.Tensor:
    batch, length, channels = x.shape
    if length < kernel:
        return x
    x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x_rm = ttnn.reshape(x_rm, (1, 1, batch * length, channels))
    out = ttnn.max_pool2d(
        input_tensor=x_rm,
        batch_size=batch,
        input_h=length,
        input_w=1,
        channels=channels,
        kernel_size=[kernel, 1],
        stride=[stride, 1],
        padding=[padding, 0],
        dilation=[1, 1],
        ceil_mode=False,
        dtype=dtype,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    out_len = max(1, (length + 2 * padding - kernel) // stride + 1)
    return ttnn.reshape(out, (batch, out_len, channels))


def apply_dropout(x: ttnn.Tensor, p: float) -> ttnn.Tensor:
    if p == 0.0:
        return x
    raise NotImplementedError("Dropout is not implemented for TTNN bring-up; set dropout=0.0.")


def prob_sparse_mask(
    q: ttnn.Tensor,
    *,
    top_u: int,
    valid_length: int,
    mask_value: float,
) -> ttnn.Tensor:
    device = q.device()
    batch, heads, length, _ = q.shape
    q_norm = ttnn.sum(q * q, dim=-1)
    idx_range = ttnn.arange(end=length, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    idx_range = ttnn.reshape(idx_range, (1, 1, length))
    valid = ttnn.lt(idx_range, valid_length, dtype=ttnn.bfloat16)
    inv_valid = ttnn.mul(valid, -1.0)
    inv_valid = inv_valid + 1.0
    q_norm = q_norm + inv_valid * mask_value
    _, topk_idx = ttnn.topk(q_norm, k=top_u, dim=-1, largest=True, sorted=False)
    topk_idx = ttnn.to_layout(topk_idx, ttnn.TILE_LAYOUT)
    topk_idx = ttnn.typecast(topk_idx, ttnn.uint32)
    topk_idx = ttnn.to_layout(topk_idx, ttnn.ROW_MAJOR_LAYOUT)
    topk_idx = ttnn.reshape(topk_idx, (batch, heads, top_u, 1))
    idx_range = ttnn.reshape(idx_range, (1, 1, 1, length))
    eq = ttnn.eq(topk_idx, idx_range, dtype=ttnn.bfloat16)
    eq = ttnn.sum(eq, dim=2)
    eq = ttnn.to_layout(eq, ttnn.ROW_MAJOR_LAYOUT)
    mask = ttnn.gt(eq, 0.0, dtype=ttnn.bfloat16)
    mask = ttnn.reshape(mask, (batch, heads, length, 1))
    return mask


def expand_mask(mask: ttnn.Tensor, head_dim: int) -> ttnn.Tensor:
    mask = ttnn.repeat(mask, (1, 1, 1, head_dim))
    return ttnn.to_layout(mask, ttnn.TILE_LAYOUT)


class ValueEmbedding:
    def __init__(self, input_dim: int, d_model: int, rng: torch.Generator, *, device, dtype: ttnn.DataType):
        self.weight_torch = torch.randn((d_model, input_dim), generator=rng, dtype=torch.float32) * 0.02
        self.bias_torch = torch.zeros((d_model,), dtype=torch.float32)
        self.weight = ttnn.from_torch(self.weight_torch, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.bias = ttnn.from_torch(self.bias_torch, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.dtype = dtype

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return linear(x, self.weight, self.bias, dtype=self.dtype)


class TemporalEmbedding:
    def __init__(self, input_dim: int, d_model: int, rng: torch.Generator, *, device, dtype: ttnn.DataType):
        self.weight_torch = torch.randn((d_model, input_dim), generator=rng, dtype=torch.float32) * 0.02
        self.bias_torch = torch.zeros((d_model,), dtype=torch.float32)
        self.weight = ttnn.from_torch(self.weight_torch, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.bias = ttnn.from_torch(self.bias_torch, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.dtype = dtype

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return linear(x, self.weight, self.bias, dtype=self.dtype)


class PositionalEmbedding:
    def __init__(self, max_len: int, d_model: int, *, device, dtype: ttnn.DataType):
        self.pe = sinusoidal_position_encoding(max_len, d_model, device=device, dtype=dtype)

    def __call__(self, length: int) -> ttnn.Tensor:
        return slice_to_length(self.pe, dim=1, length=length)


class InformerEmbedding:
    def __init__(self, config: InformerConfig, rng: torch.Generator, *, device):
        self.config = config
        self.dtype = get_ttnn_dtype(config.dtype)
        self.value_embedding = ValueEmbedding(config.enc_in, config.d_model, rng, device=device, dtype=self.dtype)
        self.temporal_embedding = TemporalEmbedding(
            config.time_feature_dim, config.d_model, rng, device=device, dtype=self.dtype
        )
        max_len = config.seq_len + config.pred_len + config.label_len
        self.positional_embedding = PositionalEmbedding(max_len, config.d_model, device=device, dtype=self.dtype)
        self.dropout = config.dropout

    def encoder(self, values: ttnn.Tensor, time_features: ttnn.Tensor) -> ttnn.Tensor:
        pos = self.positional_embedding(values.shape[1])
        x = self.value_embedding(values) + self.temporal_embedding(time_features) + pos
        return apply_dropout(x, self.dropout)

    def decoder(self, values: ttnn.Tensor, time_features: ttnn.Tensor) -> ttnn.Tensor:
        pos = self.positional_embedding(values.shape[1])
        x = self.value_embedding(values) + self.temporal_embedding(time_features) + pos
        return apply_dropout(x, self.dropout)


class MultiHeadAttention:
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        rng: torch.Generator,
        *,
        device,
        dtype: ttnn.DataType,
        prob_sparse: bool = False,
        factor: int = 5,
        mask_value: float = -1e4,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.prob_sparse = prob_sparse
        self.factor = factor
        self.mask_value = mask_value
        self.dtype = dtype

        self.q_weight_torch = torch.randn((d_model, d_model), generator=rng, dtype=torch.float32) * 0.02
        self.k_weight_torch = torch.randn((d_model, d_model), generator=rng, dtype=torch.float32) * 0.02
        self.v_weight_torch = torch.randn((d_model, d_model), generator=rng, dtype=torch.float32) * 0.02
        self.o_weight_torch = torch.randn((d_model, d_model), generator=rng, dtype=torch.float32) * 0.02
        self.q_bias_torch = torch.zeros((d_model,), dtype=torch.float32)
        self.k_bias_torch = torch.zeros((d_model,), dtype=torch.float32)
        self.v_bias_torch = torch.zeros((d_model,), dtype=torch.float32)
        self.o_bias_torch = torch.zeros((d_model,), dtype=torch.float32)

        self.q_weight = ttnn.from_torch(self.q_weight_torch, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.k_weight = ttnn.from_torch(self.k_weight_torch, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.v_weight = ttnn.from_torch(self.v_weight_torch, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.o_weight = ttnn.from_torch(self.o_weight_torch, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.q_bias = ttnn.from_torch(self.q_bias_torch, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.k_bias = ttnn.from_torch(self.k_bias_torch, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.v_bias = ttnn.from_torch(self.v_bias_torch, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.o_bias = ttnn.from_torch(self.o_bias_torch, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    def _split_heads(self, x: ttnn.Tensor, *, batch: int, length: int) -> ttnn.Tensor:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (batch, length, self.n_heads, self.head_dim))
        x = ttnn.transpose(x, 1, 2)
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def _merge_heads(self, x: ttnn.Tensor, *, batch: int, length: int) -> ttnn.Tensor:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.transpose(x, 1, 2)
        x = ttnn.reshape(x, (batch, length, self.d_model))
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def __call__(
        self,
        query_in: ttnn.Tensor,
        key_in: ttnn.Tensor,
        value_in: ttnn.Tensor,
        mask: ttnn.Tensor | None,
        *,
        q_valid_len: int | None = None,
        k_valid_len: int | None = None,
    ) -> ttnn.Tensor:
        batch, q_len, _ = query_in.shape
        _, k_len, _ = key_in.shape
        if q_valid_len is None:
            q_valid_len = q_len
        if k_valid_len is None:
            k_valid_len = k_len

        query_rm = ttnn.to_layout(query_in, ttnn.ROW_MAJOR_LAYOUT)
        key_rm = ttnn.to_layout(key_in, ttnn.ROW_MAJOR_LAYOUT)
        value_rm = ttnn.to_layout(value_in, ttnn.ROW_MAJOR_LAYOUT)

        query_rm, _ = pad_to_multiple(query_rm, dim=1, multiple=TILE_SIZE, value=0.0)
        key_rm, _ = pad_to_multiple(key_rm, dim=1, multiple=TILE_SIZE, value=0.0)
        value_rm, _ = pad_to_multiple(value_rm, dim=1, multiple=TILE_SIZE, value=0.0)

        query_in = ttnn.to_layout(query_rm, ttnn.TILE_LAYOUT)
        key_in = ttnn.to_layout(key_rm, ttnn.TILE_LAYOUT)
        value_in = ttnn.to_layout(value_rm, ttnn.TILE_LAYOUT)

        q = linear(query_in, self.q_weight, self.q_bias, dtype=self.dtype)
        k = linear(key_in, self.k_weight, self.k_bias, dtype=self.dtype)
        v = linear(value_in, self.v_weight, self.v_bias, dtype=self.dtype)

        qh = self._split_heads(q, batch=batch, length=q.shape[1])
        kh = self._split_heads(k, batch=batch, length=k.shape[1])
        vh = self._split_heads(v, batch=batch, length=v.shape[1])

        kh_t = ttnn.transpose(kh, -2, -1)
        attn_scores = ttnn.matmul(qh, kh_t)

        key_padding_mask = make_padding_mask(
            valid_length=k_valid_len,
            key_length=k.shape[1],
            batch=batch,
            heads=1,
            device=qh.device(),
            dtype=self.dtype,
            mask_value=self.mask_value,
        )
        if mask is not None:
            if mask.shape[0] != batch:
                mask = ttnn.repeat(mask, (batch // mask.shape[0], 1, 1, 1))
            attn_mask = mask + key_padding_mask
        else:
            attn_mask = key_padding_mask

        if attn_mask is not None and attn_mask.shape[2] > 1:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_scores = attn_scores * scale
            attn_scores = attn_scores + attn_mask
            attn_probs = ttnn.softmax(attn_scores, dim=-1, numeric_stable=True)
        else:
            attn_probs = ttnn.transformer.attention_softmax_(
                attn_scores,
                attention_mask=attn_mask,
                head_size=self.head_dim,
            )

        context = ttnn.matmul(attn_probs, vh)

        if self.prob_sparse:
            u = max(1, int(self.factor * math.log(max(2, q_valid_len))))
            mask_q = prob_sparse_mask(qh, top_u=u, valid_length=q_valid_len, mask_value=self.mask_value)
            mask_q = expand_mask(mask_q, self.head_dim)
            context = ttnn.mul(context, mask_q)

        context = self._merge_heads(context, batch=batch, length=q.shape[1])
        context = slice_to_length(context, dim=1, length=q_len)
        out = linear(context, self.o_weight, self.o_bias, dtype=self.dtype)
        out = apply_dropout(out, self.dropout)
        return out


class LayerNorm:
    def __init__(self, d_model: int, rng: torch.Generator, *, device, dtype: ttnn.DataType, eps: float = 1e-5):
        self.weight_torch = torch.ones((d_model,), dtype=torch.float32)
        self.bias_torch = torch.zeros((d_model,), dtype=torch.float32)
        weight = self.weight_torch.reshape(1, -1)
        bias = self.bias_torch.reshape(1, -1)
        self.weight = ttnn.from_torch(weight, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.bias = ttnn.from_torch(bias, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.eps = eps

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.layer_norm(x, weight=self.weight, bias=self.bias, epsilon=self.eps)


class FeedForward:
    def __init__(self, d_model: int, d_ff: int, dropout: float, rng: torch.Generator, *, device, dtype: ttnn.DataType):
        self.w1_torch = torch.randn((d_ff, d_model), generator=rng, dtype=torch.float32) * 0.02
        self.b1_torch = torch.zeros((d_ff,), dtype=torch.float32)
        self.w2_torch = torch.randn((d_model, d_ff), generator=rng, dtype=torch.float32) * 0.02
        self.b2_torch = torch.zeros((d_model,), dtype=torch.float32)
        self.w1 = ttnn.from_torch(self.w1_torch, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.b1 = ttnn.from_torch(self.b1_torch, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.w2 = ttnn.from_torch(self.w2_torch, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.b2 = ttnn.from_torch(self.b2_torch, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.dropout = dropout
        self.dtype = dtype

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = linear(x, self.w1, self.b1, dtype=self.dtype)
        x = ttnn.gelu(x)
        x = apply_dropout(x, self.dropout)
        x = linear(x, self.w2, self.b2, dtype=self.dtype)
        x = apply_dropout(x, self.dropout)
        return x


class EncoderLayer:
    def __init__(self, config: InformerConfig, rng: torch.Generator, *, device, dtype: ttnn.DataType):
        self.attn = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            rng=rng,
            device=device,
            dtype=dtype,
            prob_sparse=True,
            factor=config.factor,
            mask_value=config.attn_mask_value,
        )
        self.ffn = FeedForward(config.d_model, config.d_ff, config.dropout, rng, device=device, dtype=dtype)
        self.norm1 = LayerNorm(config.d_model, rng, device=device, dtype=dtype)
        self.norm2 = LayerNorm(config.d_model, rng, device=device, dtype=dtype)

    def __call__(self, x: ttnn.Tensor, mask: ttnn.Tensor | None, valid_length: int) -> ttnn.Tensor:
        attn_out = self.attn(x, x, x, mask, q_valid_len=valid_length, k_valid_len=valid_length)
        x = self.norm1(x + attn_out)
        ff = self.ffn(x)
        x = self.norm2(x + ff)
        return x


class Encoder:
    def __init__(self, config: InformerConfig, rng: torch.Generator, *, device):
        self.dtype = get_ttnn_dtype(config.dtype)
        self.layers = [EncoderLayer(config, rng, device=device, dtype=self.dtype) for _ in range(config.e_layers)]
        self.distil = config.distil
        self.distil_norm = (
            LayerNorm(config.d_model, rng, device=device, dtype=self.dtype) if self.distil.enabled else None
        )
        self.mask_value = config.attn_mask_value

    def __call__(self, x: ttnn.Tensor, mask: ttnn.Tensor | None) -> tuple[ttnn.Tensor, int]:
        valid_length = x.shape[1]
        for i, layer in enumerate(self.layers):
            x = layer(x, mask, valid_length)
            if self.distil.enabled and i < len(self.layers) - 1:
                x = max_pool1d(
                    x,
                    kernel=self.distil.kernel_size,
                    stride=self.distil.stride,
                    padding=self.distil.padding,
                    dtype=self.dtype,
                )
                valid_length = max(
                    1,
                    (valid_length + 2 * self.distil.padding - self.distil.kernel_size) // self.distil.stride + 1,
                )
                x, _ = pad_to_multiple(x, dim=1, multiple=TILE_SIZE, value=0.0)
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
                x = self.distil_norm(x)
        return x, valid_length


class DecoderLayer:
    def __init__(self, config: InformerConfig, rng: torch.Generator, *, device, dtype: ttnn.DataType):
        self.self_attn = MultiHeadAttention(
            config.d_model,
            config.n_heads,
            config.dropout,
            rng,
            device=device,
            dtype=dtype,
            prob_sparse=False,
            mask_value=config.attn_mask_value,
        )
        self.cross_attn = MultiHeadAttention(
            config.d_model,
            config.n_heads,
            config.dropout,
            rng,
            device=device,
            dtype=dtype,
            prob_sparse=False,
            mask_value=config.attn_mask_value,
        )
        self.ffn = FeedForward(config.d_model, config.d_ff, config.dropout, rng, device=device, dtype=dtype)
        self.norm1 = LayerNorm(config.d_model, rng, device=device, dtype=dtype)
        self.norm2 = LayerNorm(config.d_model, rng, device=device, dtype=dtype)
        self.norm3 = LayerNorm(config.d_model, rng, device=device, dtype=dtype)

    def __call__(
        self,
        x: ttnn.Tensor,
        enc_out: ttnn.Tensor,
        self_mask: ttnn.Tensor | None,
        cross_mask: ttnn.Tensor | None,
        enc_valid_len: int,
    ) -> ttnn.Tensor:
        attn1 = self.self_attn(x, x, x, self_mask, q_valid_len=x.shape[1], k_valid_len=x.shape[1])
        x = self.norm1(x + attn1)

        attn2 = self.cross_attn(x, enc_out, enc_out, cross_mask, q_valid_len=x.shape[1], k_valid_len=enc_valid_len)
        x = self.norm2(x + attn2)

        ff = self.ffn(x)
        x = self.norm3(x + ff)
        return x


class Decoder:
    def __init__(self, config: InformerConfig, rng: torch.Generator, *, device):
        self.dtype = get_ttnn_dtype(config.dtype)
        self.layers = [DecoderLayer(config, rng, device=device, dtype=self.dtype) for _ in range(config.d_layers)]

    def __call__(
        self,
        x: ttnn.Tensor,
        enc_out: ttnn.Tensor,
        self_mask: ttnn.Tensor | None,
        cross_mask: ttnn.Tensor | None,
        enc_valid_len: int,
    ) -> ttnn.Tensor:
        for layer in self.layers:
            x = layer(x, enc_out, self_mask, cross_mask, enc_valid_len)
        return x


class InformerModel:
    """
    Informer time-series forecasting model implemented in TTNN.

    The model implements the full Informer architecture:
    - Value/temporal/positional embeddings
    - ProbSparse encoder with distilling
    - Generative decoder with masked self-attention and cross-attention
    - Projection head for predictions
    """

    def __init__(self, config: InformerConfig, *, device, seed: int = 0):
        self.config = config
        self.device = device
        self.dtype = get_ttnn_dtype(config.dtype)
        self.rng = torch.Generator().manual_seed(seed)
        self.embedding = InformerEmbedding(config, self.rng, device=device)
        self.encoder = Encoder(config, self.rng, device=device)
        self.decoder = Decoder(config, self.rng, device=device)
        self.proj_w_torch = torch.randn((config.c_out, config.d_model), generator=self.rng, dtype=torch.float32) * 0.02
        self.proj_b_torch = torch.zeros((config.c_out,), dtype=torch.float32)
        self.proj_w = ttnn.from_torch(self.proj_w_torch, device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
        self.proj_b = ttnn.from_torch(self.proj_b_torch, device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)

    def _to_device(self, x: torch.Tensor | ttnn.Tensor) -> ttnn.Tensor:
        if isinstance(x, ttnn.Tensor):
            return x
        return ttnn.from_torch(x, device=self.device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Export model weights as a state dictionary."""
        state: dict[str, torch.Tensor] = {}
        emb = self.embedding
        state["embedding.value.weight"] = emb.value_embedding.weight_torch.clone()
        state["embedding.value.bias"] = emb.value_embedding.bias_torch.clone()
        state["embedding.temporal.weight"] = emb.temporal_embedding.weight_torch.clone()
        state["embedding.temporal.bias"] = emb.temporal_embedding.bias_torch.clone()

        for i, layer in enumerate(self.encoder.layers):
            prefix = f"encoder.layers.{i}"
            attn = layer.attn
            state[f"{prefix}.attn.q_weight"] = attn.q_weight_torch.clone()
            state[f"{prefix}.attn.q_bias"] = attn.q_bias_torch.clone()
            state[f"{prefix}.attn.k_weight"] = attn.k_weight_torch.clone()
            state[f"{prefix}.attn.k_bias"] = attn.k_bias_torch.clone()
            state[f"{prefix}.attn.v_weight"] = attn.v_weight_torch.clone()
            state[f"{prefix}.attn.v_bias"] = attn.v_bias_torch.clone()
            state[f"{prefix}.attn.o_weight"] = attn.o_weight_torch.clone()
            state[f"{prefix}.attn.o_bias"] = attn.o_bias_torch.clone()

            state[f"{prefix}.ffn.w1"] = layer.ffn.w1_torch.clone()
            state[f"{prefix}.ffn.b1"] = layer.ffn.b1_torch.clone()
            state[f"{prefix}.ffn.w2"] = layer.ffn.w2_torch.clone()
            state[f"{prefix}.ffn.b2"] = layer.ffn.b2_torch.clone()

            state[f"{prefix}.norm1.weight"] = layer.norm1.weight_torch.clone()
            state[f"{prefix}.norm1.bias"] = layer.norm1.bias_torch.clone()
            state[f"{prefix}.norm2.weight"] = layer.norm2.weight_torch.clone()
            state[f"{prefix}.norm2.bias"] = layer.norm2.bias_torch.clone()

        if self.encoder.distil_norm is not None:
            state["encoder.distil_norm.weight"] = self.encoder.distil_norm.weight_torch.clone()
            state["encoder.distil_norm.bias"] = self.encoder.distil_norm.bias_torch.clone()

        for i, layer in enumerate(self.decoder.layers):
            prefix = f"decoder.layers.{i}"
            attn = layer.self_attn
            state[f"{prefix}.self_attn.q_weight"] = attn.q_weight_torch.clone()
            state[f"{prefix}.self_attn.q_bias"] = attn.q_bias_torch.clone()
            state[f"{prefix}.self_attn.k_weight"] = attn.k_weight_torch.clone()
            state[f"{prefix}.self_attn.k_bias"] = attn.k_bias_torch.clone()
            state[f"{prefix}.self_attn.v_weight"] = attn.v_weight_torch.clone()
            state[f"{prefix}.self_attn.v_bias"] = attn.v_bias_torch.clone()
            state[f"{prefix}.self_attn.o_weight"] = attn.o_weight_torch.clone()
            state[f"{prefix}.self_attn.o_bias"] = attn.o_bias_torch.clone()

            attn = layer.cross_attn
            state[f"{prefix}.cross_attn.q_weight"] = attn.q_weight_torch.clone()
            state[f"{prefix}.cross_attn.q_bias"] = attn.q_bias_torch.clone()
            state[f"{prefix}.cross_attn.k_weight"] = attn.k_weight_torch.clone()
            state[f"{prefix}.cross_attn.k_bias"] = attn.k_bias_torch.clone()
            state[f"{prefix}.cross_attn.v_weight"] = attn.v_weight_torch.clone()
            state[f"{prefix}.cross_attn.v_bias"] = attn.v_bias_torch.clone()
            state[f"{prefix}.cross_attn.o_weight"] = attn.o_weight_torch.clone()
            state[f"{prefix}.cross_attn.o_bias"] = attn.o_bias_torch.clone()

            state[f"{prefix}.ffn.w1"] = layer.ffn.w1_torch.clone()
            state[f"{prefix}.ffn.b1"] = layer.ffn.b1_torch.clone()
            state[f"{prefix}.ffn.w2"] = layer.ffn.w2_torch.clone()
            state[f"{prefix}.ffn.b2"] = layer.ffn.b2_torch.clone()

            state[f"{prefix}.norm1.weight"] = layer.norm1.weight_torch.clone()
            state[f"{prefix}.norm1.bias"] = layer.norm1.bias_torch.clone()
            state[f"{prefix}.norm2.weight"] = layer.norm2.weight_torch.clone()
            state[f"{prefix}.norm2.bias"] = layer.norm2.bias_torch.clone()
            state[f"{prefix}.norm3.weight"] = layer.norm3.weight_torch.clone()
            state[f"{prefix}.norm3.bias"] = layer.norm3.bias_torch.clone()

        state["projection.weight"] = self.proj_w_torch.clone()
        state["projection.bias"] = self.proj_b_torch.clone()
        return state

    def load_state_dict(self, state: dict[str, torch.Tensor], *, strict: bool = True) -> dict[str, list[str]]:
        """Load model weights from a state dictionary."""
        expected = set(self.state_dict().keys())
        provided = set(state.keys())
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        if strict and (missing or unexpected):
            raise ValueError(f"State dict mismatch. Missing: {missing}. Unexpected: {unexpected}.")

        def to_float(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.detach().float()

        def upload_tensor(tensor: torch.Tensor) -> ttnn.Tensor:
            return ttnn.from_torch(tensor, device=self.device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)

        def update_embedding(emb, prefix: str):
            weight_key = f"{prefix}.weight"
            bias_key = f"{prefix}.bias"
            if weight_key in state:
                value = to_float(state[weight_key])
                emb.weight_torch = value
                emb.weight = upload_tensor(value)
            if bias_key in state:
                value = to_float(state[bias_key])
                emb.bias_torch = value
                emb.bias = upload_tensor(value)

        def update_attention(attn, prefix: str):
            for name in ("q_weight", "q_bias", "k_weight", "k_bias", "v_weight", "v_bias", "o_weight", "o_bias"):
                key = f"{prefix}.{name}"
                if key not in state:
                    continue
                value = to_float(state[key])
                setattr(attn, f"{name}_torch", value)
                setattr(attn, name, upload_tensor(value))

        def update_ffn(ffn, prefix: str):
            for name in ("w1", "b1", "w2", "b2"):
                key = f"{prefix}.{name}"
                if key not in state:
                    continue
                value = to_float(state[key])
                setattr(ffn, f"{name}_torch", value)
                setattr(ffn, name, upload_tensor(value))

        def update_norm(norm, prefix: str):
            weight_key = f"{prefix}.weight"
            bias_key = f"{prefix}.bias"
            if weight_key in state:
                value = to_float(state[weight_key])
                norm.weight_torch = value
                norm.weight = upload_tensor(value.reshape(1, -1))
            if bias_key in state:
                value = to_float(state[bias_key])
                norm.bias_torch = value
                norm.bias = upload_tensor(value.reshape(1, -1))

        update_embedding(self.embedding.value_embedding, "embedding.value")
        update_embedding(self.embedding.temporal_embedding, "embedding.temporal")

        for i, layer in enumerate(self.encoder.layers):
            prefix = f"encoder.layers.{i}"
            update_attention(layer.attn, f"{prefix}.attn")
            update_ffn(layer.ffn, f"{prefix}.ffn")
            update_norm(layer.norm1, f"{prefix}.norm1")
            update_norm(layer.norm2, f"{prefix}.norm2")

        if self.encoder.distil_norm is not None:
            update_norm(self.encoder.distil_norm, "encoder.distil_norm")

        for i, layer in enumerate(self.decoder.layers):
            prefix = f"decoder.layers.{i}"
            update_attention(layer.self_attn, f"{prefix}.self_attn")
            update_attention(layer.cross_attn, f"{prefix}.cross_attn")
            update_ffn(layer.ffn, f"{prefix}.ffn")
            update_norm(layer.norm1, f"{prefix}.norm1")
            update_norm(layer.norm2, f"{prefix}.norm2")
            update_norm(layer.norm3, f"{prefix}.norm3")

        if "projection.weight" in state:
            value = to_float(state["projection.weight"])
            self.proj_w_torch = value
            self.proj_w = upload_tensor(value)
        if "projection.bias" in state:
            value = to_float(state["projection.bias"])
            self.proj_b_torch = value
            self.proj_b = upload_tensor(value)

        return {"missing_keys": missing, "unexpected_keys": unexpected}

    def __call__(
        self,
        past_values: torch.Tensor | ttnn.Tensor,
        past_time_features: torch.Tensor | ttnn.Tensor,
        future_time_features: torch.Tensor | ttnn.Tensor,
        future_values: torch.Tensor | ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """
        Forward pass of the Informer model.

        Args:
            past_values: Historical time series values [batch, seq_len, features]
            past_time_features: Time features for past values [batch, seq_len, time_dim]
            future_time_features: Time features for future values [batch, pred_len, time_dim]
            future_values: Optional future values for teacher forcing [batch, pred_len, features]

        Returns:
            Predictions for the forecast horizon [batch, pred_len, c_out]
        """
        cfg = self.config
        past_values = self._to_device(past_values)
        past_time_features = self._to_device(past_time_features)
        future_time_features = self._to_device(future_time_features)
        future_values = self._to_device(future_values) if future_values is not None else None

        enc_embed = self.embedding.encoder(past_values, past_time_features)
        enc_out, enc_valid_len = self.encoder(enc_embed, None)

        label_len = cfg.label_len
        dec_known = past_values[:, -label_len:, :]
        if future_values is None:
            future_pad = ttnn.zeros(
                (past_values.shape[0], cfg.pred_len, cfg.dec_in),
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
        else:
            future_pad = future_values[:, : cfg.pred_len, :]
        dec_values = ttnn.concat([dec_known, future_pad], dim=1)

        dec_time = ttnn.concat(
            [past_time_features[:, -label_len:, :], future_time_features[:, : cfg.pred_len, :]], dim=1
        )
        dec_embed = self.embedding.decoder(dec_values, dec_time)

        dec_len = dec_embed.shape[1]
        pad_len = int(math.ceil(dec_len / TILE_SIZE)) * TILE_SIZE
        self_mask = make_causal_mask(
            pad_len,
            batch=past_values.shape[0],
            heads=1,
            device=self.device,
            dtype=self.dtype,
            mask_value=cfg.attn_mask_value,
        )
        dec_out = self.decoder(dec_embed, enc_out, self_mask, None, enc_valid_len)

        output = linear(dec_out[:, -cfg.pred_len :, :], self.proj_w, self.proj_b, dtype=self.dtype)
        return output

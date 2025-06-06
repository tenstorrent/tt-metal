# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import ttnn

from .linear import TtLinear, TtLinearParameters
from .substate import indexed_substates, substate
from .utils import from_torch_fast


@dataclass
class TtT5Config:
    vocab_size: int
    d_model: int
    d_ff: int
    d_kv: int
    num_heads: int
    num_layers: int
    relative_attention_num_buckets: int
    relative_attention_max_distance: int
    layer_norm_epsilon: float


@dataclass
class TtT5EncoderParameters:
    token_embedding: ttnn.Tensor
    blocks: [TtT5BlockParameters]
    norm: TtT5LayerNorm
    attention_bias: ttnn.Tensor

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtT5EncoderParameters:
        return cls(
            token_embedding=from_torch_fast(
                state["encoder.embed_tokens.weight"],
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=dtype,
                device=device,
                shard_dim=None,
            ),
            blocks=[
                TtT5BlockParameters.from_torch(s, dtype=dtype, device=device)
                for s in indexed_substates(state, "encoder.block")
            ],
            norm=TtT5LayerNormParameters.from_torch(
                substate(state, "encoder.final_layer_norm"), dtype=dtype, device=device
            ),
            attention_bias=from_torch_fast(
                state["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"],
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=dtype,
                device=device,
                shard_dim=None,
            ),
        )


class TtT5Encoder:
    def __init__(
        self,
        parameters: TtT5EncoderParameters,
        num_heads: int,
        relative_attention_num_buckets: int,
        relative_attention_max_distance: int,
        layer_norm_epsilon: float,
    ) -> None:
        self._token_embedding = parameters.token_embedding
        self._blocks = [
            TtT5Block(p, num_heads=num_heads, layer_norm_epsilon=layer_norm_epsilon) for p in parameters.blocks
        ]
        self._norm = TtT5LayerNorm(parameters.norm, eps=layer_norm_epsilon)
        self._attention_bias = parameters.attention_bias
        self._num_heads = num_heads
        self._relative_attention_num_buckets = relative_attention_num_buckets
        self._relative_attention_max_distance = relative_attention_max_distance

    def __call__(self, input_ids: ttnn.Tensor, device) -> ttnn.Tensor:
        _batch_size, seq_length = input_ids.shape

        # TODO: Remove the conversion to row major layout once ttnn.embedding works with tiled input
        # https://github.com/tenstorrent/tt-metal/issues/17643
        input_ids = ttnn.to_layout(input_ids, ttnn.ROW_MAJOR_LAYOUT)
        inputs_embeds = ttnn.embedding(input_ids, self._token_embedding, layout=ttnn.TILE_LAYOUT)

        position_bias = _compute_bias(
            seq_length=seq_length,
            device=device,
            relative_attention_num_buckets=self._relative_attention_num_buckets,
            relative_attention_max_distance=self._relative_attention_max_distance,
            relative_attention_bias=self._attention_bias,
        )

        x = inputs_embeds
        for block in self._blocks:
            x = block(x, position_bias=position_bias)

        return self._norm(x)


@dataclass
class TtT5BlockParameters:
    attention: TtT5LayerSelfAttentionParameters
    ff: TtT5LayerFFParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtT5BlockParameters:
        return cls(
            attention=TtT5LayerSelfAttentionParameters.from_torch(
                substate(state, "layer.0"), dtype=dtype, device=device
            ),
            ff=TtT5LayerFFParameters.from_torch(substate(state, "layer.1"), dtype=dtype, device=device),
        )


class TtT5Block:
    def __init__(self, parameters: TtT5BlockParameters, *, num_heads: int, layer_norm_epsilon: float) -> None:
        self._attention = TtT5LayerSelfAttention(
            parameters.attention, num_heads=num_heads, layer_norm_epsilon=layer_norm_epsilon
        )
        self._ff = TtT5LayerFF(parameters.ff, layer_norm_epsilon=layer_norm_epsilon)

    def __call__(self, x: ttnn.Tensor, *, position_bias: ttnn.Tensor) -> ttnn.Tensor:
        x = self._attention(x, position_bias=position_bias)
        return self._ff(x)


@dataclass
class TtT5LayerSelfAttentionParameters:
    attention: TtT5AttentionParameters
    norm: TtT5LayerNormParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtT5LayerSelfAttentionParameters:
        return cls(
            attention=TtT5AttentionParameters.from_torch(substate(state, "SelfAttention"), dtype=dtype, device=device),
            norm=TtT5LayerNormParameters.from_torch(substate(state, "layer_norm"), dtype=dtype, device=device),
        )


class TtT5LayerSelfAttention:
    def __init__(self, parameters: TtT5BlockParameters, *, num_heads: int, layer_norm_epsilon: float) -> None:
        self._attention = TtT5Attention(parameters.attention, num_heads=num_heads)
        self._norm = TtT5LayerNorm(parameters.norm, eps=layer_norm_epsilon)

    def __call__(self, x: ttnn.Tensor, *, position_bias: ttnn.Tensor) -> ttnn.Tensor:
        normed = self._norm(x)
        attn = self._attention(normed, position_bias=position_bias)
        return x + attn


@dataclass
class TtT5AttentionParameters:
    q_proj: ttnn.Tensor
    k_proj: ttnn.Tensor
    v_proj: ttnn.Tensor
    o_proj: ttnn.Tensor

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtT5AttentionParameters:
        return cls(
            q_proj=TtLinearParameters.from_torch(substate(state, "q"), dtype=dtype, device=device, shard_dim=None),
            k_proj=TtLinearParameters.from_torch(substate(state, "k"), dtype=dtype, device=device, shard_dim=None),
            v_proj=TtLinearParameters.from_torch(substate(state, "v"), dtype=dtype, device=device, shard_dim=None),
            o_proj=TtLinearParameters.from_torch(substate(state, "o"), dtype=dtype, device=device, shard_dim=None),
        )


class TtT5Attention:
    def __init__(self, parameters: TtT5AttentionParameters, *, num_heads: int) -> None:
        self._num_heads = num_heads

        self._q_proj = TtLinear(parameters.q_proj)
        self._k_proj = TtLinear(parameters.k_proj)
        self._v_proj = TtLinear(parameters.v_proj)
        self._o_proj = TtLinear(parameters.o_proj)

    def __call__(self, x: ttnn.Tensor, *, position_bias: ttnn.Tensor) -> ttnn.Tensor:
        batch_size, seq_length, _ = x.shape

        q = self._q_proj(x)
        k = self._k_proj(x)
        v = self._v_proj(x)

        qkv = ttnn.concat([q, k, v], dim=-1)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=self._num_heads, transpose_key=True
        )
        scores = ttnn.matmul(q, k)
        scores = scores + position_bias
        attn_weights = ttnn.softmax(scores, dim=-1)
        attn = ttnn.matmul(attn_weights, v)
        attn = ttnn.transformer.concatenate_heads(attn)

        return self._o_proj(attn)


@dataclass
class TtT5LayerFFParameters:
    dense_gated_dense: TtT5DenseGatedActDenseParameters
    norm: TtT5LayerNormParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtT5LayerFFParameters:
        return cls(
            dense_gated_dense=TtT5DenseGatedActDenseParameters.from_torch(
                substate(state, "DenseReluDense"), dtype=dtype, device=device
            ),
            norm=TtT5LayerNormParameters.from_torch(substate(state, "layer_norm"), dtype=dtype, device=device),
        )


class TtT5LayerFF:
    def __init__(self, parameters: TtT5LayerFFParameters, *, layer_norm_epsilon: float) -> None:
        self._dense_gated_dense = TtT5DenseGatedActDense(parameters.dense_gated_dense)
        self._norm = TtT5LayerNorm(parameters.norm, eps=layer_norm_epsilon)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        fw = self._norm(x)
        fw = self._dense_gated_dense(fw)
        return x + fw


@dataclass
class TtT5DenseGatedActDenseParameters:
    wi0: TtLinearParameters
    wi1: TtLinearParameters
    wo: TtLinearParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtT5DenseGatedActDenseParameters:
        return cls(
            wi0=TtLinearParameters.from_torch(substate(state, "wi_0"), dtype=dtype, device=device, shard_dim=None),
            wi1=TtLinearParameters.from_torch(substate(state, "wi_1"), dtype=dtype, device=device, shard_dim=None),
            wo=TtLinearParameters.from_torch(substate(state, "wo"), dtype=dtype, device=device, shard_dim=None),
        )


class TtT5DenseGatedActDense:
    def __init__(self, parameters: TtT5DenseGatedActDenseParameters) -> None:
        self._wi0 = TtLinear(parameters.wi0)
        self._wi1 = TtLinear(parameters.wi1)
        self._wo = TtLinear(parameters.wo)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gelu = new_gelu_activation(self._wi0(x))
        linear = self._wi1(x)
        x = gelu * linear
        return self._wo(x)


@dataclass
class TtT5LayerNormParameters:
    weight: ttnn.Tensor

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtT5LayerNormParameters:
        return cls(
            weight=from_torch_fast(state["weight"], layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device),
        )


class TtT5LayerNorm:
    def __init__(self, parameters: TtT5LayerNormParameters, *, eps: float) -> None:
        self._weight = parameters.weight
        self._eps = eps

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        variance = ttnn.mean(ttnn.pow(x, 2), -1, keepdim=True)
        x *= ttnn.rsqrt(variance + self._eps)
        return self._weight * x


def _relative_position_bucket(relative_position: torch.Tensor, num_buckets: int, max_distance: int) -> torch.Tensor:
    num_buckets //= 2

    relative_buckets = (relative_position > 0).to(torch.long) * num_buckets
    relative_position = torch.abs(relative_position)

    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1),
    )

    relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets


def _compute_bias(
    *,
    seq_length: int,
    device: ttnn.Device,
    relative_attention_num_buckets: int,
    relative_attention_max_distance: int,
    relative_attention_bias: ttnn.Tensor,
) -> ttnn.Tensor:
    context_position = torch.arange(seq_length)[:, None]
    memory_position = torch.arange(seq_length)[None, :]
    relative_position = memory_position - context_position

    relative_position_bucket = _relative_position_bucket(
        relative_position,
        num_buckets=relative_attention_num_buckets,
        max_distance=relative_attention_max_distance,
    )

    relative_attention_bias = ttnn.get_device_tensors(relative_attention_bias)[0]
    torch_relative_attention_bias = ttnn.to_torch(relative_attention_bias)
    # torch_relative_attention_bias = to_torch(relative_attention_bias, mesh_device=device, dtype=relative_attention_bias.get_dtype(), shard_dim=None)
    output = torch.nn.functional.embedding(relative_position_bucket, torch_relative_attention_bias)
    output = output.permute([2, 0, 1]).unsqueeze(0)
    output = output[:, :, -seq_length:, :]

    return from_torch_fast(
        output, device=device, dtype=relative_attention_bias.get_dtype(), shard_dim=None, layout=ttnn.TILE_LAYOUT
    )


def new_gelu_activation(x: ttnn.Tensor) -> ttnn.Tensor:
    c = math.sqrt(2.0 / math.pi)
    y = 0.044715 * ttnn.pow(x, 3) + x
    return 0.5 * x * (1.0 + ttnn.tanh(c * y))

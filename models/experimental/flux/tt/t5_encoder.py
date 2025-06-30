# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import ttnn

from .linear import Linear, LinearParameters
from .substate import indexed_substates, substate
from .utils import from_torch_fast


@dataclass
class T5Config:
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
class T5EncoderParameters:
    token_embedding: ttnn.Tensor
    blocks: list[T5BlockParameters]
    norm: T5LayerNorm
    attention_bias: torch.Tensor
    device: ttnn.MeshDevice

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
    ) -> T5EncoderParameters:
        return cls(
            token_embedding=from_torch_fast(
                state["encoder.embed_tokens.weight"],
                dtype=dtype,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(device, tuple(device.shape), (None, -1)),
            ),
            blocks=[
                T5BlockParameters.from_torch(s, dtype=dtype, device=device)
                for s in indexed_substates(state, "encoder.block")
            ],
            norm=T5LayerNormParameters.from_torch(
                substate(state, "encoder.final_layer_norm"), dtype=dtype, device=device
            ),
            attention_bias=state["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"],
            device=device,
        )


class T5Encoder:
    def __init__(
        self,
        parameters: T5EncoderParameters,
        num_heads: int,
        relative_attention_num_buckets: int,
        relative_attention_max_distance: int,
        layer_norm_epsilon: float,
    ) -> None:
        self._token_embedding = parameters.token_embedding
        self._blocks = [
            T5Block(p, num_heads=num_heads, layer_norm_epsilon=layer_norm_epsilon) for p in parameters.blocks
        ]
        self._norm = T5LayerNorm(parameters.norm, eps=layer_norm_epsilon)
        self._attention_bias = parameters.attention_bias

        self._num_heads = num_heads
        self._relative_attention_num_buckets = relative_attention_num_buckets
        self._relative_attention_max_distance = relative_attention_max_distance

        self._device = parameters.device

    def forward(self, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        _batch_size, seq_length = input_ids.shape

        # TODO: Remove the conversion to row major layout once ttnn.embedding works with tiled input
        # https://github.com/tenstorrent/tt-metal/issues/17643
        input_ids = ttnn.to_layout(input_ids, ttnn.ROW_MAJOR_LAYOUT)
        inputs_embeds = ttnn.embedding(input_ids, self._token_embedding, layout=ttnn.TILE_LAYOUT)

        position_bias = _compute_bias(
            seq_length=seq_length,
            device=self._device,
            relative_attention_num_buckets=self._relative_attention_num_buckets,
            relative_attention_max_distance=self._relative_attention_max_distance,
            relative_attention_bias=self._attention_bias,
        )

        x = inputs_embeds
        for block in self._blocks:
            x = block.forward(x, position_bias=position_bias)

        return self._norm.forward(x)


@dataclass
class T5BlockParameters:
    attention: T5LayerSelfAttentionParameters
    ff: T5LayerFFParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
    ) -> T5BlockParameters:
        return cls(
            attention=T5LayerSelfAttentionParameters.from_torch(substate(state, "layer.0"), dtype=dtype, device=device),
            ff=T5LayerFFParameters.from_torch(substate(state, "layer.1"), dtype=dtype, device=device),
        )


class T5Block:
    def __init__(self, parameters: T5BlockParameters, *, num_heads: int, layer_norm_epsilon: float) -> None:
        self._attention = T5LayerSelfAttention(
            parameters.attention, num_heads=num_heads, layer_norm_epsilon=layer_norm_epsilon
        )
        self._ff = T5LayerFF(parameters.ff, layer_norm_epsilon=layer_norm_epsilon)

    def forward(self, x: ttnn.Tensor, *, position_bias: ttnn.Tensor) -> ttnn.Tensor:
        x = self._attention.forward(x, position_bias=position_bias)
        return self._ff.forward(x)


@dataclass
class T5LayerSelfAttentionParameters:
    attention: T5AttentionParameters
    norm: T5LayerNormParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
    ) -> T5LayerSelfAttentionParameters:
        return cls(
            attention=T5AttentionParameters.from_torch(substate(state, "SelfAttention"), dtype=dtype, device=device),
            norm=T5LayerNormParameters.from_torch(substate(state, "layer_norm"), dtype=dtype, device=device),
        )


class T5LayerSelfAttention:
    def __init__(self, parameters: T5BlockParameters, *, num_heads: int, layer_norm_epsilon: float) -> None:
        self._attention = T5Attention(parameters.attention, num_heads=num_heads)
        self._norm = T5LayerNorm(parameters.norm, eps=layer_norm_epsilon)

    def forward(self, x: ttnn.Tensor, *, position_bias: ttnn.Tensor) -> ttnn.Tensor:
        normed = self._norm.forward(x)
        attn = self._attention.forward(normed, position_bias=position_bias)
        return x + attn


@dataclass
class T5AttentionParameters:
    q_proj: ttnn.Tensor
    k_proj: ttnn.Tensor
    v_proj: ttnn.Tensor
    o_proj: ttnn.Tensor
    mesh_width: int

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
    ) -> T5AttentionParameters:
        _, mesh_width = device.shape

        common = dict(
            dtype=dtype,
            device=device,
            mesh_sharding_dim=0,
        )

        return cls(
            q_proj=LinearParameters.from_torch(substate(state, "q"), **common),
            k_proj=LinearParameters.from_torch(substate(state, "k"), **common),
            v_proj=LinearParameters.from_torch(substate(state, "v"), **common),
            o_proj=LinearParameters.from_torch(substate(state, "o"), **common),
            mesh_width=mesh_width,
        )


class T5Attention:
    def __init__(self, parameters: T5AttentionParameters, *, num_heads: int) -> None:
        self._num_heads = num_heads
        self._mesh_width = parameters.mesh_width

        self._q_proj = Linear(parameters.q_proj)
        self._k_proj = Linear(parameters.k_proj)
        self._v_proj = Linear(parameters.v_proj)
        self._o_proj = Linear(parameters.o_proj)

    def forward(self, x: ttnn.Tensor, *, position_bias: ttnn.Tensor) -> ttnn.Tensor:
        q = self._q_proj.forward(x)
        k = self._k_proj.forward(x)
        v = self._v_proj.forward(x)

        qkv = ttnn.concat([q, k, v], dim=-1)

        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=self._num_heads // self._mesh_width, transpose_key=True
        )

        scores = ttnn.matmul(q, k) + position_bias
        attn_weights = ttnn.softmax(scores, dim=-1)
        attn = ttnn.matmul(attn_weights, v)
        attn = ttnn.transformer.concatenate_heads(attn)

        return self._o_proj.forward(attn)


@dataclass
class T5LayerFFParameters:
    dense_gated_dense: T5DenseGatedActDenseParameters
    norm: T5LayerNormParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
    ) -> T5LayerFFParameters:
        return cls(
            dense_gated_dense=T5DenseGatedActDenseParameters.from_torch(
                substate(state, "DenseReluDense"), dtype=dtype, device=device
            ),
            norm=T5LayerNormParameters.from_torch(substate(state, "layer_norm"), dtype=dtype, device=device),
        )


class T5LayerFF:
    def __init__(self, parameters: T5LayerFFParameters, *, layer_norm_epsilon: float) -> None:
        self._dense_gated_dense = T5DenseGatedActDense(parameters.dense_gated_dense)
        self._norm = T5LayerNorm(parameters.norm, eps=layer_norm_epsilon)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        fw = self._norm.forward(x)
        fw = self._dense_gated_dense.forward(fw)
        return x + fw


@dataclass
class T5DenseGatedActDenseParameters:
    wi0: LinearParameters
    wi1: LinearParameters
    wo: LinearParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
    ) -> T5DenseGatedActDenseParameters:
        common = dict(
            dtype=dtype,
            device=device,
            mesh_sharding_dim=0,
        )

        return cls(
            wi0=LinearParameters.from_torch(substate(state, "wi_0"), **common),
            wi1=LinearParameters.from_torch(substate(state, "wi_1"), **common),
            wo=LinearParameters.from_torch(substate(state, "wo"), **common),
        )


class T5DenseGatedActDense:
    def __init__(self, parameters: T5DenseGatedActDenseParameters) -> None:
        self._wi0 = Linear(parameters.wi0)
        self._wi1 = Linear(parameters.wi1)
        self._wo = Linear(parameters.wo)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gelu = new_gelu_activation(self._wi0.forward(x))
        linear = self._wi1.forward(x)
        x = gelu * linear
        return self._wo.forward(x)


@dataclass
class T5LayerNormParameters:
    weight: ttnn.Tensor

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
    ) -> T5LayerNormParameters:
        return cls(
            weight=from_torch_fast(
                state["weight"],
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                device=device,
                mesh_mapper=ttnn.ShardTensor2dMesh(device, tuple(device.shape), (None, -1)),
            ),
        )


class T5LayerNorm:
    def __init__(self, parameters: T5LayerNormParameters, *, eps: float) -> None:
        self._weight = parameters.weight
        self._eps = eps

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
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
    device: ttnn.MeshDevice,
    relative_attention_num_buckets: int,
    relative_attention_max_distance: int,
    relative_attention_bias: torch.Tensor,
) -> ttnn.Tensor:
    context_position = torch.arange(seq_length)[:, None]
    memory_position = torch.arange(seq_length)[None, :]
    relative_position = memory_position - context_position

    relative_position_bucket = _relative_position_bucket(
        relative_position,
        num_buckets=relative_attention_num_buckets,
        max_distance=relative_attention_max_distance,
    )

    output = torch.nn.functional.embedding(relative_position_bucket, relative_attention_bias)
    output = output.permute([2, 0, 1]).unsqueeze(0)
    output = output[:, :, -seq_length:, :]

    return from_torch_fast(
        output,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, tuple(device.shape), (None, 1)),
    )


def new_gelu_activation(x: ttnn.Tensor) -> ttnn.Tensor:
    c = math.sqrt(2.0 / math.pi)
    y = 0.044715 * ttnn.pow(x, 3) + x
    return 0.5 * x * (1.0 + ttnn.tanh(c * y, accuracy=True))

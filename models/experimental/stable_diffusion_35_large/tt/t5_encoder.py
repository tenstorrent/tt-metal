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
from .parallel_config import EncoderParallelManager


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
        parallel_manager: EncoderParallelManager,
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
                TtT5BlockParameters.from_torch(s, dtype=dtype, device=device, parallel_manager=parallel_manager)
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

    def __call__(
        self, input_ids: ttnn.Tensor, device: ttnn.Device, parallel_manager: EncoderParallelManager
    ) -> ttnn.Tensor:
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
            parallel_manager=parallel_manager,
        )

        x = inputs_embeds
        for block in self._blocks:
            x = block(x, position_bias=position_bias, parallel_manager=parallel_manager)

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
        parallel_manager: EncoderParallelManager,
    ) -> TtT5BlockParameters:
        return cls(
            attention=TtT5LayerSelfAttentionParameters.from_torch(
                substate(state, "layer.0"), dtype=dtype, device=device, parallel_manager=parallel_manager
            ),
            ff=TtT5LayerFFParameters.from_torch(
                substate(state, "layer.1"), dtype=dtype, device=device, parallel_manager=parallel_manager
            ),
        )


class TtT5Block:
    def __init__(self, parameters: TtT5BlockParameters, *, num_heads: int, layer_norm_epsilon: float) -> None:
        self._attention = TtT5LayerSelfAttention(
            parameters.attention, num_heads=num_heads, layer_norm_epsilon=layer_norm_epsilon
        )
        self._ff = TtT5LayerFF(parameters.ff, layer_norm_epsilon=layer_norm_epsilon)

    def __call__(
        self, x: ttnn.Tensor, *, position_bias: ttnn.Tensor, parallel_manager: EncoderParallelManager
    ) -> ttnn.Tensor:
        x = self._attention(x, position_bias=position_bias, parallel_manager=parallel_manager)
        return self._ff(x, parallel_manager)


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
        parallel_manager: EncoderParallelManager,
    ) -> TtT5LayerSelfAttentionParameters:
        return cls(
            attention=TtT5AttentionParameters.from_torch(
                substate(state, "SelfAttention"), dtype=dtype, device=device, parallel_manager=parallel_manager
            ),
            norm=TtT5LayerNormParameters.from_torch(substate(state, "layer_norm"), dtype=dtype, device=device),
        )


class TtT5LayerSelfAttention:
    def __init__(self, parameters: TtT5BlockParameters, *, num_heads: int, layer_norm_epsilon: float) -> None:
        self._attention = TtT5Attention(parameters.attention, num_heads=num_heads)
        self._norm = TtT5LayerNorm(parameters.norm, eps=layer_norm_epsilon)

    def __call__(
        self, x: ttnn.Tensor, *, position_bias: ttnn.Tensor, parallel_manager: EncoderParallelManager
    ) -> ttnn.Tensor:
        normed = self._norm(x)
        attn = self._attention(normed, position_bias=position_bias, parallel_manager=parallel_manager)
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
        parallel_manager: EncoderParallelManager,
    ) -> TtT5AttentionParameters:
        def column_parallel_linear(name):
            my_state = substate(state, name)
            weight = my_state["weight"]  # [num_heads * head_dim, input_dim]
            bias = my_state.get("bias", None)  # [num_heads * head_dim]
            weight = weight.T  # [input_dim, num_heads * head_dim]

            shard_dims = [None, None]
            shard_dims[parallel_manager.tensor_parallel.mesh_axis] = -1

            weight = ttnn.from_torch(
                weight,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=shard_dims),
            )

            if bias is not None:
                bias = bias.unsqueeze(0)
                bias = ttnn.from_torch(
                    bias,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=shard_dims),
                )

            return TtLinearParameters(weight=weight, bias=bias)

        q_proj = column_parallel_linear("q")
        k_proj = column_parallel_linear("k")
        v_proj = column_parallel_linear("v")
        o_proj = column_parallel_linear("o")

        return cls(
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            o_proj=o_proj,
        )


class TtT5Attention:
    def __init__(self, parameters: TtT5AttentionParameters, *, num_heads: int) -> None:
        self._num_heads = num_heads

        self._q_proj = TtLinear(parameters.q_proj)
        self._k_proj = TtLinear(parameters.k_proj)
        self._v_proj = TtLinear(parameters.v_proj)
        self._o_proj = TtLinear(parameters.o_proj)

    def __call__(
        self, x: ttnn.Tensor, *, position_bias: ttnn.Tensor, parallel_manager: EncoderParallelManager
    ) -> ttnn.Tensor:
        batch_size, seq_length, _ = x.shape

        q = self._q_proj(x)
        k = self._k_proj(x)
        v = self._v_proj(x)

        qkv = ttnn.concat([q, k, v], dim=-1)
        # reshape for multihead attention
        num_devices = parallel_manager.tensor_parallel.factor
        num_local_heads = self._num_heads // num_devices
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=num_local_heads, transpose_key=True
        )
        scores = ttnn.matmul(q, k)
        scores = scores + position_bias
        attn_weights = ttnn.softmax(scores, dim=-1)
        attn = ttnn.matmul(attn_weights, v)
        attn = ttnn.transformer.concatenate_heads(attn)

        attn = ttnn.unsqueeze(attn, 0)
        orig_shape = list(attn.shape)
        attn = ttnn.experimental.all_gather_async(
            attn,
            dim=len(attn.shape) - 1,
            cluster_axis=parallel_manager.tensor_parallel.mesh_axis,
            mesh_device=parallel_manager.mesh_device,
            topology=parallel_manager.topology,
            multi_device_global_semaphore=parallel_manager.get_ping_pong_semaphore(),
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        dense_out = self._o_proj(attn)

        dense_out = ttnn.experimental.all_gather_async(
            dense_out,
            dim=len(dense_out.shape) - 1,
            cluster_axis=parallel_manager.tensor_parallel.mesh_axis,
            mesh_device=parallel_manager.mesh_device,
            topology=parallel_manager.topology,
            multi_device_global_semaphore=parallel_manager.get_ping_pong_semaphore(),
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        dense_out_shape = list(dense_out.shape)
        dense_out_shape[2] = orig_shape[2]
        dense_out = ttnn.reshape(dense_out, tuple(dense_out_shape), dense_out.shape)

        return ttnn.reshape(dense_out, tuple(dense_out.shape)[1:])


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
        parallel_manager: EncoderParallelManager,
    ) -> TtT5LayerFFParameters:
        return cls(
            dense_gated_dense=TtT5DenseGatedActDenseParameters.from_torch(
                substate(state, "DenseReluDense"), dtype=dtype, device=device, parallel_manager=parallel_manager
            ),
            norm=TtT5LayerNormParameters.from_torch(substate(state, "layer_norm"), dtype=dtype, device=device),
        )


class TtT5LayerFF:
    def __init__(self, parameters: TtT5LayerFFParameters, *, layer_norm_epsilon: float) -> None:
        self._dense_gated_dense = TtT5DenseGatedActDense(parameters.dense_gated_dense)
        self._norm = TtT5LayerNorm(parameters.norm, eps=layer_norm_epsilon)

    def __call__(self, x: ttnn.Tensor, parallel_manager: EncoderParallelManager) -> ttnn.Tensor:
        fw = self._norm(x)
        fw = self._dense_gated_dense(fw, parallel_manager)
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
        parallel_manager: EncoderParallelManager,
    ) -> TtT5DenseGatedActDenseParameters:
        def parallel_weight_bias(name, dim):
            assert dim in [-1, -2]
            my_state = substate(state, name)
            weight = my_state["weight"].transpose(0, 1)
            bias = my_state.get("bias", None)

            weight_shard_dims = [None, None]
            weight_shard_dims[parallel_manager.tensor_parallel.mesh_axis] = dim

            weight = ttnn.from_torch(
                weight,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=weight_shard_dims),
            )
            if bias is not None:
                # Bias always sharded on last dimension. If row-parallel, extend
                # bias with zeros.
                bias_shard_dims = [None, None]
                bias_shard_dims[parallel_manager.tensor_parallel.mesh_axis] = -1
                bias = bias.unsqueeze(0)
                if dim == -2:
                    # row-parallel, only one device should apply bias
                    zero_bias = torch.zeros_like(bias)
                    bias = torch.cat([bias] + [zero_bias] * (parallel_manager.tensor_parallel.factor - 1), dim=-1)
                bias = ttnn.from_torch(
                    bias,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=bias_shard_dims),
                )
            return TtLinearParameters(weight=weight, bias=bias)

        return cls(
            wi0=parallel_weight_bias("wi_0", -1),
            wi1=parallel_weight_bias("wi_1", -1),
            wo=parallel_weight_bias("wo", -2),
        )


class TtT5DenseGatedActDense:
    def __init__(self, parameters: TtT5DenseGatedActDenseParameters) -> None:
        self._wi0 = TtLinear(parameters.wi0)
        self._wi1 = TtLinear(parameters.wi1)
        self._wo = TtLinear(parameters.wo)

    def __call__(self, x: ttnn.Tensor, parallel_manager: EncoderParallelManager) -> ttnn.Tensor:
        gelu = new_gelu_activation(self._wi0(x))
        linear = self._wi1(x)
        x = gelu * linear
        hidden_states = self._wo(x)

        hidden_states_shape = list(hidden_states.shape)
        hidden_states = ttnn.unsqueeze(hidden_states, 0)
        # AllReduce output

        hidden_states_scattered = ttnn.experimental.reduce_scatter_minimal_async(
            hidden_states,
            dim=3,
            multi_device_global_semaphore=parallel_manager.get_rs_ping_pong_semaphore(),
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=parallel_manager.topology,
            cluster_axis=parallel_manager.tensor_parallel.mesh_axis,
        )
        hidden_states = ttnn.experimental.all_gather_async(
            hidden_states_scattered,
            dim=3,
            cluster_axis=parallel_manager.tensor_parallel.mesh_axis,
            mesh_device=parallel_manager.mesh_device,
            topology=parallel_manager.topology,
            multi_device_global_semaphore=parallel_manager.get_ping_pong_semaphore(),
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden_states = ttnn.reshape(hidden_states, hidden_states_shape, hidden_states.shape)
        return hidden_states


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
    parallel_manager: EncoderParallelManager,
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
    # Shard outputs on dim=-3, heads
    shard_dims = [None, None]
    shard_dims[parallel_manager.tensor_parallel.mesh_axis] = -3
    return ttnn.from_torch(
        output,
        device=device,
        dtype=relative_attention_bias.get_dtype(),
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=shard_dims),
    )


def new_gelu_activation(x: ttnn.Tensor) -> ttnn.Tensor:
    c = math.sqrt(2.0 / math.pi)
    y = 0.044715 * ttnn.pow(x, 3) + x
    return 0.5 * x * (1.0 + ttnn.tanh(c * y))

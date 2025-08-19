# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn
from ttnn.distributed.distributed import ConcatMeshToTensor

from .linear import TtLinear, TtLinearParameters
from .substate import indexed_substates, substate
from .utils import from_torch_fast
from .parallel_config import EncoderParallelManager
import math


@dataclass
class CLIPEncoderOutput:
    # stores all hidden states
    hidden_states: list[ttnn.Tensor]

    def __getitem__(self, idx):
        return self.hidden_states[idx]


@dataclass
class TtCLIPConfig:
    vocab_size: int
    d_model: int  # embedding dim
    d_ff: int  # mlp dim
    num_heads: int
    num_layers: int  # num transformer blocks
    max_position_embeddings: int
    layer_norm_eps: float
    attention_dropout: float
    hidden_act: str


@dataclass
class TtCLIPAttentionParameters:
    q_proj: TtLinearParameters
    k_proj: TtLinearParameters
    v_proj: TtLinearParameters
    o_proj: TtLinearParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
        parallel_manager: EncoderParallelManager = None,
    ) -> TtCLIPAttentionParameters:
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

        if parallel_manager is not None:
            q_proj = column_parallel_linear("q_proj")
            k_proj = column_parallel_linear("k_proj")
            v_proj = column_parallel_linear("v_proj")
            o_proj = column_parallel_linear("out_proj")
        else:
            # Plain data parallelism
            q_proj = TtLinearParameters.from_torch(substate(state, "q_proj"), dtype=dtype, device=device)
            k_proj = TtLinearParameters.from_torch(substate(state, "k_proj"), dtype=dtype, device=device)
            v_proj = TtLinearParameters.from_torch(substate(state, "v_proj"), dtype=dtype, device=device)
            o_proj = TtLinearParameters.from_torch(substate(state, "out_proj"), dtype=dtype, device=device)

        return cls(
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            o_proj=o_proj,
        )


class TtCLIPAttention:
    def __init__(
        self,
        parameters: TtCLIPAttentionParameters,
        config: TtCLIPConfig,
    ) -> None:
        self._num_heads = config.num_heads
        self._embed_dim = config.d_model
        self._head_dim = config.d_model // config.num_heads
        self._scale = self._head_dim**-0.5
        self._attention_dropout = config.attention_dropout

        self._q_proj = TtLinear(parameters.q_proj)
        self._k_proj = TtLinear(parameters.k_proj)
        self._v_proj = TtLinear(parameters.v_proj)
        self._o_proj = TtLinear(parameters.o_proj)

    def __call__(
        self, hidden_states: ttnn.Tensor, causal_mask: ttnn.Tensor, parallel_manager: EncoderParallelManager = None
    ) -> ttnn.Tensor:
        """
        In cases of parallel_manager valid:

        input is replicated
        Q, K, V are head-parallel
        SDPA executes head-parallel
        output is replicated

        Else assume plain data parallelism, all inputs are split over batch and weights are replicated.
        """
        batch_size, seq_length, _ = hidden_states.shape

        q = self._q_proj(hidden_states)  # head_parallel
        k = self._k_proj(hidden_states)  # head_parallel
        v = self._v_proj(hidden_states)  # head_parallel

        q = q * self._scale

        # reshape for multihead attention
        if parallel_manager is not None:
            num_devices = parallel_manager.tensor_parallel.factor
            num_local_heads = self._num_heads // num_devices
        else:
            num_devices = 1
            num_local_heads = self._num_heads

        q = ttnn.reshape(q, (batch_size, seq_length, num_local_heads, self._head_dim))
        k = ttnn.reshape(k, (batch_size, seq_length, num_local_heads, self._head_dim))
        v = ttnn.reshape(v, (batch_size, seq_length, num_local_heads, self._head_dim))

        # transpose to [batch_size, num_heads, seq_length, head_dim]
        q = ttnn.transpose(q, 1, 2)
        k = ttnn.transpose(k, 1, 2)
        v = ttnn.transpose(v, 1, 2)

        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))

        if causal_mask is not None:
            scores = scores + causal_mask

        attn_weights = ttnn.softmax(scores, dim=-1)

        # TODO: replace with ttnn.dropout once it's supported
        # attn_weights = ttnn.experimental.dropout(attn_weights, self._attention_dropout)

        attn_output = ttnn.matmul(attn_weights, v)  # head_parallel

        # transpose back and reshape
        attn_output = ttnn.transpose(attn_output, 1, 2)
        attn_output = ttnn.reshape(attn_output, (1, batch_size, seq_length, self._embed_dim // num_devices))

        # all-gather
        orig_shape = list(attn_output.shape)
        if parallel_manager is not None:
            attn_output = ttnn.experimental.all_gather_async(
                attn_output,
                dim=len(attn_output.shape) - 1,
                cluster_axis=parallel_manager.tensor_parallel.mesh_axis,
                mesh_device=parallel_manager.mesh_device,
                topology=parallel_manager.topology,
                multi_device_global_semaphore=parallel_manager.get_ping_pong_semaphore(),
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        dense_out = self._o_proj(attn_output)

        if parallel_manager is not None:
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
class TtCLIPMLPParameters:
    fc1: TtLinearParameters
    fc2: TtLinearParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
        parallel_manager: EncoderParallelManager = None,
    ) -> TtCLIPMLPParameters:
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
                bias = bias.unsqueeze(0)
                if dim == -2:
                    # row-parallel, only one device should apply bias
                    zero_bias = torch.zeros_like(bias)
                    bias = torch.cat([bias] + [zero_bias] * (parallel_manager.tensor_parallel.factor - 1), dim=-1)

                bias_shard_dims = [None, None]
                bias_shard_dims[parallel_manager.tensor_parallel.mesh_axis] = -1
                bias = ttnn.from_torch(
                    bias,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=bias_shard_dims),
                )
            return TtLinearParameters(weight=weight, bias=bias)

        fc1 = (
            parallel_weight_bias("fc1", -1)
            if parallel_manager is not None
            else TtLinearParameters.from_torch(substate(state, "fc1"), dtype=dtype, device=device)
        )
        fc2 = (
            parallel_weight_bias("fc2", -2)
            if parallel_manager is not None
            else TtLinearParameters.from_torch(substate(state, "fc2"), dtype=dtype, device=device)
        )
        return cls(
            fc1=fc1,
            fc2=fc2,
        )


class TtCLIPMLP:
    def __init__(self, parameters: TtCLIPMLPParameters, config: TtCLIPConfig) -> None:
        self._fc1 = TtLinear(parameters.fc1)
        self._fc2 = TtLinear(parameters.fc2)
        self._hidden_act = config.hidden_act

    def __call__(self, hidden_states: ttnn.Tensor, parallel_manager: EncoderParallelManager = None) -> ttnn.Tensor:
        hidden_states = self._fc1(hidden_states)

        if self._hidden_act == "gelu":
            hidden_states = gelu(hidden_states)  # HF default gelu
        else:  # quick approx gelu
            hidden_states = hidden_states * ttnn.sigmoid(1.702 * hidden_states)

        hidden_states = self._fc2(hidden_states)
        if parallel_manager is not None:
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


def gelu(x: ttnn.Tensor) -> ttnn.Tensor:
    # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    # ttnn.gelu is the same, but avoiding for potential issues (see ttnn.layernorm)
    sqrt_2 = math.sqrt(2.0)
    x_div_sqrt2 = ttnn.multiply(x, 1.0 / sqrt_2)
    erf_x = ttnn.erf(x_div_sqrt2)
    one_plus_erf = ttnn.add(erf_x, 1.0)
    x_times_bracket = ttnn.multiply(x, one_plus_erf)
    return ttnn.multiply(x_times_bracket, 0.5)


@dataclass
class TtCLIPEncoderLayerParameters:
    self_attn: TtCLIPAttentionParameters
    mlp: TtCLIPMLPParameters
    layer_norm1_weight: ttnn.Tensor
    layer_norm1_bias: ttnn.Tensor
    layer_norm2_weight: ttnn.Tensor
    layer_norm2_bias: ttnn.Tensor

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
        parallel_manager: EncoderParallelManager = None,
    ) -> TtCLIPEncoderLayerParameters:
        return cls(
            self_attn=TtCLIPAttentionParameters.from_torch(
                substate(state, "self_attn"), dtype=dtype, device=device, parallel_manager=parallel_manager
            ),
            mlp=TtCLIPMLPParameters.from_torch(
                substate(state, "mlp"), dtype=dtype, device=device, parallel_manager=parallel_manager
            ),
            layer_norm1_weight=from_torch_fast(
                state["layer_norm1.weight"], dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT
            ),
            layer_norm1_bias=from_torch_fast(
                state["layer_norm1.bias"], dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT
            ),
            layer_norm2_weight=from_torch_fast(
                state["layer_norm2.weight"], dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT
            ),
            layer_norm2_bias=from_torch_fast(
                state["layer_norm2.bias"], dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT
            ),
        )


class TtCLIPEncoderLayer:
    def __init__(
        self,
        parameters: TtCLIPEncoderLayerParameters,
        config: TtCLIPConfig,
    ) -> None:
        self._self_attn = TtCLIPAttention(parameters.self_attn, config)
        self._mlp = TtCLIPMLP(parameters.mlp, config)
        self._layer_norm1 = parameters.layer_norm1_weight
        self._layer_norm1_bias = parameters.layer_norm1_bias
        self._layer_norm2 = parameters.layer_norm2_weight
        self._layer_norm2_bias = parameters.layer_norm2_bias
        self._layer_norm_eps = config.layer_norm_eps

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        causal_attention_mask: ttnn.Tensor,
        parallel_manager: EncoderParallelManager = None,
    ) -> ttnn.Tensor:
        # self attention block
        residual = hidden_states
        hidden_states = ttnn.layer_norm(
            hidden_states, weight=self._layer_norm1, bias=self._layer_norm1_bias, epsilon=self._layer_norm_eps
        )
        attn_output = self._self_attn(hidden_states, causal_attention_mask, parallel_manager=parallel_manager)
        hidden_states = residual + attn_output

        # mlp block
        residual = hidden_states
        hidden_states = ttnn.layer_norm(
            hidden_states, weight=self._layer_norm2, bias=self._layer_norm2_bias, epsilon=self._layer_norm_eps
        )
        mlp_output = self._mlp(hidden_states, parallel_manager=parallel_manager)
        hidden_states = residual + mlp_output

        return hidden_states


@dataclass
class TtCLIPTransformerParameters:
    layers: list[TtCLIPEncoderLayerParameters]

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
        parallel_manager: EncoderParallelManager = None,
    ) -> TtCLIPTransformerParameters:
        layers = []
        layer_states = indexed_substates(state, "layers")
        for layer_state in layer_states:
            layers.append(
                TtCLIPEncoderLayerParameters.from_torch(
                    layer_state, dtype=dtype, device=device, parallel_manager=parallel_manager
                )
            )

        return cls(
            layers=layers,
        )


class TtCLIPTransformer:
    def __init__(self, parameters: TtCLIPTransformerParameters, config: TtCLIPConfig) -> None:
        self._config = config
        self._layers = [TtCLIPEncoderLayer(layer_params, config) for layer_params in parameters.layers]

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        causal_attention_mask: ttnn.Tensor,
        parallel_manager: EncoderParallelManager = None,
        output_hidden_states: bool = True,
    ) -> CLIPEncoderOutput:
        all_hidden_states = []

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        for layer in self._layers:
            hidden_states = layer(hidden_states, causal_attention_mask, parallel_manager=parallel_manager)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        return CLIPEncoderOutput(hidden_states=all_hidden_states)


@dataclass
class TtCLIPEncoderParameters:
    text_model: TtCLIPTransformerParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
        parallel_manager: EncoderParallelManager = None,
    ) -> TtCLIPEncoderParameters:
        return cls(
            text_model=TtCLIPTransformerParameters.from_torch(
                state,
                dtype=dtype,
                device=device,
                parallel_manager=parallel_manager,
            )
        )


class TtCLIPEncoder:
    def __init__(self, parameters: TtCLIPEncoderParameters, config: TtCLIPConfig) -> None:
        self._text_model = TtCLIPTransformer(parameters.text_model, config)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        causal_attention_mask: ttnn.Tensor,
        parallel_manager: EncoderParallelManager = None,
        output_hidden_states: bool = True,
    ) -> CLIPEncoderOutput:
        return self._text_model(
            hidden_states,
            causal_attention_mask,
            parallel_manager=parallel_manager,
            output_hidden_states=output_hidden_states,
        )


@dataclass
class TtCLIPEmbeddingParameters:
    token_embedding: ttnn.Tensor
    position_embedding: ttnn.Tensor

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtCLIPEmbeddingParameters:
        # weights must be bfloat16 for ttnn.embedding ops
        embedding_dtype = ttnn.bfloat16
        return cls(
            token_embedding=from_torch_fast(
                state["token_embedding.weight"],
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=embedding_dtype,
                device=device,
            ),
            position_embedding=from_torch_fast(
                state["position_embedding.weight"],
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=embedding_dtype,
                device=device,
            ),
        )


class TtCLIPEmbedding:
    def __init__(self, parameters: TtCLIPEmbeddingParameters, config: TtCLIPConfig) -> None:
        self._token_embedding = parameters.token_embedding
        self._position_embedding = parameters.position_embedding
        self._max_position_embeddings = config.max_position_embeddings

    def __call__(self, input_ids: ttnn.Tensor, device: ttnn.Device) -> ttnn.Tensor:
        seq_length = input_ids.shape[-1]

        # truncate seq if >max_position_embeddings
        if seq_length > self._max_position_embeddings:
            input_ids = input_ids[:, : self._max_position_embeddings]
            seq_length = self._max_position_embeddings

        position_ids = torch.arange(seq_length).expand((1, -1))
        position_ids = ttnn.from_torch(position_ids, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)

        input_embeddings = ttnn.embedding(input_ids, self._token_embedding, layout=ttnn.TILE_LAYOUT)
        position_embeddings = ttnn.embedding(position_ids, self._position_embedding, layout=ttnn.TILE_LAYOUT)

        return input_embeddings + position_embeddings


@dataclass
class TtCLIPTextTransformerParameters:
    embeddings: TtCLIPEmbeddingParameters
    encoder: TtCLIPEncoderParameters
    final_layer_norm_weight: ttnn.Tensor
    final_layer_norm_bias: ttnn.Tensor
    text_projection_weight: ttnn.Tensor

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
        parallel_manager: EncoderParallelManager = None,
        has_text_projection: bool = True,  # SDXL Base encoder 1 does not have text projection, so this is a quick hack to handle it
    ) -> TtCLIPTextTransformerParameters:
        text_model_state = substate(state, "text_model")

        return cls(
            embeddings=TtCLIPEmbeddingParameters.from_torch(
                substate(text_model_state, "embeddings"),
                dtype=dtype,
                device=device,
            ),
            encoder=TtCLIPEncoderParameters.from_torch(
                substate(text_model_state, "encoder"), dtype=dtype, device=device, parallel_manager=parallel_manager
            ),
            final_layer_norm_weight=from_torch_fast(
                text_model_state["final_layer_norm.weight"], dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT
            ),
            final_layer_norm_bias=from_torch_fast(
                text_model_state["final_layer_norm.bias"], dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT
            ),
            text_projection_weight=from_torch_fast(
                state["text_projection.weight"], dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT
            )
            if has_text_projection
            else None,
        )


class TtCLIPTextTransformer:
    def __init__(
        self,
        parameters: TtCLIPTextTransformerParameters,
        config: TtCLIPConfig,
    ) -> None:
        self._config = config
        self._embeddings = TtCLIPEmbedding(parameters.embeddings, config)
        self._encoder = TtCLIPEncoder(parameters.encoder, config)
        self._final_layer_norm = parameters.final_layer_norm_weight
        self._final_layer_norm_bias = parameters.final_layer_norm_bias
        self._layer_norm_eps = config.layer_norm_eps
        self._text_projection = parameters.text_projection_weight

    def __call__(
        self,
        input_ids: ttnn.Tensor,
        device: ttnn.Device,
        eos_token_id: int = None,
        parallel_manager: EncoderParallelManager = None,
        output_hidden_states: bool = True,
        clip_skip: int = None,
    ) -> tuple[CLIPEncoderOutput, ttnn.Tensor]:
        batch_size, seq_length = input_ids.shape

        hidden_states = self._embeddings(input_ids, device)

        causal_attention_mask = _create_4d_causal_attention_mask(
            input_ids.shape, device, dtype=hidden_states.get_dtype()
        )

        encoder_output = self._encoder(
            hidden_states,
            causal_attention_mask,
            parallel_manager=parallel_manager,
            output_hidden_states=output_hidden_states,
        )

        final_hidden_state = encoder_output.hidden_states[-1]  # Last encoder layer output
        normalized_final_state = ttnn.layer_norm(
            final_hidden_state,
            weight=self._final_layer_norm,
            bias=self._final_layer_norm_bias,
            epsilon=self._layer_norm_eps,
        )

        if eos_token_id is None:
            eos_token_id = 2

        pooled_output = self._gather_eos(normalized_final_state, input_ids, eos_token_id, device)

        if self._text_projection is not None:
            text_projection_transposed = ttnn.transpose(self._text_projection, -2, -1)
            projected_output = ttnn.matmul(pooled_output, text_projection_transposed)
        else:
            projected_output = pooled_output

        return encoder_output, projected_output

    def _pool_eos_from_torch_tensors(self, ids_t: torch.Tensor, seq_t: torch.Tensor, eos_token_id: int) -> torch.Tensor:
        """Helper function to pool EOS tokens from torch tensors.

        Args:
            ids_t: Token IDs tensor [B, S]
            seq_t: Sequence embeddings tensor [B, S, H]
            eos_token_id: EOS token ID to search for

        Returns:
            Pooled tensor [B, H]
        """
        # from HF: if self.eos_token_id == 2: use argmax, else: search for eos_token_id
        if eos_token_id == 2:
            # use argmax (highest token ID position)
            eos_idx = ids_t.to(dtype=torch.int, device=ids_t.device).argmax(dim=-1)
        else:
            # search for specific eos_token_id
            eos_mask = (ids_t.to(dtype=torch.int, device=ids_t.device) == eos_token_id).int()
            eos_idx = eos_mask.argmax(dim=-1)

        # Use vectorized indexing to get pooled output
        b = torch.arange(seq_t.size(0))
        pooled_t = seq_t[b, eos_idx]  # [B, H]
        return pooled_t

    def _gather_eos(
        self,
        seq_emb: ttnn.Tensor,
        input_ids: ttnn.Tensor,
        eos_token_id: int,
        device: ttnn.Device,
        encoder_parallel_manager: EncoderParallelManager = None,
    ) -> ttnn.Tensor:
        if encoder_parallel_manager is not None:
            ids_t = ttnn.to_torch(ttnn.get_device_tensors(input_ids)[0])
            seq_t = ttnn.to_torch(ttnn.get_device_tensors(seq_emb)[0])  # [B, S, H]

            pooled_t = self._pool_eos_from_torch_tensors(ids_t, seq_t, eos_token_id)

            return ttnn.from_torch(
                pooled_t,
                dtype=seq_emb.get_dtype(),
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
        else:
            ids_t = ttnn.to_torch(input_ids, mesh_composer=ConcatMeshToTensor(device, dim=0))
            seq_t = ttnn.to_torch(seq_emb, mesh_composer=ConcatMeshToTensor(device, dim=0))

            pooled_t = self._pool_eos_from_torch_tensors(ids_t, seq_t, eos_token_id)

            return ttnn.from_torch(
                pooled_t,
                dtype=seq_emb.get_dtype(),
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),
            )


# adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py
def _create_4d_causal_attention_mask(
    input_shape: tuple[int, int], device: ttnn.Device, dtype: ttnn.DataType
) -> ttnn.Tensor:
    """Create a 4D causal attention mask for the given input shape."""
    batch_size, tgt_len = input_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask[None, None, :, :].expand(batch_size, 1, tgt_len, tgt_len)
    return ttnn.from_torch(mask, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

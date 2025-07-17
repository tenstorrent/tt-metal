# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

from .linear import TtLinear, TtLinearParameters
from .substate import indexed_substates, substate
from .utils import from_torch_fast


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
    ) -> TtCLIPAttentionParameters:
        return cls(
            q_proj=TtLinearParameters.from_torch(substate(state, "q_proj"), dtype=dtype, device=device),
            k_proj=TtLinearParameters.from_torch(substate(state, "k_proj"), dtype=dtype, device=device),
            v_proj=TtLinearParameters.from_torch(substate(state, "v_proj"), dtype=dtype, device=device),
            o_proj=TtLinearParameters.from_torch(substate(state, "out_proj"), dtype=dtype, device=device),
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

    def __call__(self, hidden_states: ttnn.Tensor, causal_mask: ttnn.Tensor) -> ttnn.Tensor:
        batch_size, seq_length, _ = hidden_states.shape

        q = self._q_proj(hidden_states)
        k = self._k_proj(hidden_states)
        v = self._v_proj(hidden_states)

        q = q * self._scale

        # reshape for multihead attention
        q = ttnn.reshape(q, (batch_size, seq_length, self._num_heads, self._head_dim))
        k = ttnn.reshape(k, (batch_size, seq_length, self._num_heads, self._head_dim))
        v = ttnn.reshape(v, (batch_size, seq_length, self._num_heads, self._head_dim))

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

        attn_output = ttnn.matmul(attn_weights, v)

        # transpose back and reshape
        attn_output = ttnn.transpose(attn_output, 1, 2)
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_length, self._embed_dim))

        return self._o_proj(attn_output)


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
    ) -> TtCLIPMLPParameters:
        return cls(
            fc1=TtLinearParameters.from_torch(substate(state, "fc1"), dtype=dtype, device=device),
            fc2=TtLinearParameters.from_torch(substate(state, "fc2"), dtype=dtype, device=device),
        )


class TtCLIPMLP:
    def __init__(self, parameters: TtCLIPMLPParameters) -> None:
        self._fc1 = TtLinear(parameters.fc1)
        self._fc2 = TtLinear(parameters.fc2)

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        hidden_states = self._fc1(hidden_states)

        hidden_states = gelu(hidden_states)
        hidden_states = self._fc2(hidden_states)
        return hidden_states


def gelu(x: ttnn.Tensor) -> ttnn.Tensor:
    return x * ttnn.sigmoid(1.702 * x)


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
    ) -> TtCLIPEncoderLayerParameters:
        return cls(
            self_attn=TtCLIPAttentionParameters.from_torch(substate(state, "self_attn"), dtype=dtype, device=device),
            mlp=TtCLIPMLPParameters.from_torch(substate(state, "mlp"), dtype=dtype, device=device),
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
        self._mlp = TtCLIPMLP(parameters.mlp)
        self._layer_norm1 = parameters.layer_norm1_weight
        self._layer_norm1_bias = parameters.layer_norm1_bias
        self._layer_norm2 = parameters.layer_norm2_weight
        self._layer_norm2_bias = parameters.layer_norm2_bias
        self._layer_norm_eps = config.layer_norm_eps

    def __call__(self, hidden_states: ttnn.Tensor, causal_attention_mask: ttnn.Tensor) -> ttnn.Tensor:
        # self attention block
        residual = hidden_states
        hidden_states = ttnn.layer_norm(
            hidden_states, weight=self._layer_norm1, bias=self._layer_norm1_bias, epsilon=self._layer_norm_eps
        )
        attn_output = self._self_attn(hidden_states, causal_attention_mask)
        hidden_states = residual + attn_output

        # mlp block
        residual = hidden_states
        hidden_states = ttnn.layer_norm(
            hidden_states, weight=self._layer_norm2, bias=self._layer_norm2_bias, epsilon=self._layer_norm_eps
        )
        mlp_output = self._mlp(hidden_states)
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
    ) -> TtCLIPTransformerParameters:
        layers = []
        layer_states = indexed_substates(state, "layers")
        for layer_state in layer_states:
            layers.append(TtCLIPEncoderLayerParameters.from_torch(layer_state, dtype=dtype, device=device))

        return cls(
            layers=layers,
        )


class TtCLIPTransformer:
    def __init__(self, parameters: TtCLIPTransformerParameters, config: TtCLIPConfig) -> None:
        self._config = config
        self._layers = [TtCLIPEncoderLayer(layer_params, config) for layer_params in parameters.layers]

    def __call__(self, hidden_states: ttnn.Tensor, causal_attention_mask: ttnn.Tensor) -> ttnn.Tensor:
        for layer in self._layers:
            hidden_states = layer(hidden_states, causal_attention_mask)

        return hidden_states


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
    ) -> TtCLIPEncoderParameters:
        return cls(
            text_model=TtCLIPTransformerParameters.from_torch(
                state, dtype=dtype, device=device  # state is already the encoder substate
            )
        )


class TtCLIPEncoder:
    def __init__(self, parameters: TtCLIPEncoderParameters, config: TtCLIPConfig) -> None:
        self._text_model = TtCLIPTransformer(parameters.text_model, config)

    def __call__(self, hidden_states: ttnn.Tensor, causal_attention_mask: ttnn.Tensor) -> ttnn.Tensor:
        return self._text_model(hidden_states, causal_attention_mask)


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
        # weights must be BFLOAT16 for ttnn.embedding ops
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

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtCLIPTextTransformerParameters:
        text_model_state = substate(state, "text_model")

        return cls(
            embeddings=TtCLIPEmbeddingParameters.from_torch(
                substate(text_model_state, "embeddings"), dtype=dtype, device=device
            ),
            encoder=TtCLIPEncoderParameters.from_torch(
                substate(text_model_state, "encoder"), dtype=dtype, device=device
            ),
            final_layer_norm_weight=from_torch_fast(
                text_model_state["final_layer_norm.weight"], dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT
            ),
            final_layer_norm_bias=from_torch_fast(
                text_model_state["final_layer_norm.bias"], dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT
            ),
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

    def __call__(self, input_ids: ttnn.Tensor, device: ttnn.Device) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        batch_size, seq_length = input_ids.shape

        hidden_states = self._embeddings(input_ids, device)

        causal_attention_mask = _create_4d_causal_attention_mask(
            input_ids.shape, device, dtype=hidden_states.get_dtype()
        )

        hidden_states = self._encoder(hidden_states, causal_attention_mask)

        sequence_embeddings = ttnn.layer_norm(
            hidden_states, weight=self._final_layer_norm, bias=self._final_layer_norm_bias, epsilon=self._layer_norm_eps
        )
        pooled_output = sequence_embeddings[:, -1, :]

        return sequence_embeddings, pooled_output


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

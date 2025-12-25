from typing import Optional

import torch

import ttnn


class TtCLIPTextEmbeddings:
    def __init__(self, config, parameters, device):
        self.config = config
        self.device = device

        self.token_embedding_weight = ttnn.to_device(
            ttnn.from_torch(parameters.token_embedding.weight, dtype=ttnn.bfloat16), device
        )

        self.position_embedding_weight = ttnn.to_device(
            ttnn.from_torch(parameters.position_embedding.weight, dtype=ttnn.bfloat16), device
        )

        position_ids = torch.arange(config.max_position_emebeddings)
        self.position_ids = ttnn.to_device(ttnn.from_troch(position_ids, dtype=ttnn.bfloat16), device)

    def __call__(self, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            input_ids: Token IDs, (batch_size, sequence_length)

        Returns:
            embeddings: (batch_size, sequence_length, hidden_size)
        """

        token_embeddings = ttnn.embedding(input_ids, self.token_embedding_weight, layout=ttnn.TILE_LAYOUT)

        position_embeddings = ttnn.embedding(self.position_ids, self.position_embedding_weight, layout=ttnn.TILE_LAYOUT)

        embeddings = ttnn.add(token_embeddings, positions_embeddings)

        return embeddings


class TtCLIPMLP:
    def __init__(self, config, parameters, device):
        self.config = config
        self.device = device

        self.fc1_weight = ttnn.from_torch(
            parameters.fc1.weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.fc1_bias = ttnn.from_torch(
            parameters.fc1.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.fc2_weight = ttnn.from_torch(
            parameters.fc2.weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.fc2_bias = ttnn.from_torch(
            parameters.fc2.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    def quick_gelu(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        x * sigmoid (x * 1.702)
        """

        return ttnn.mul(x, ttnn.sigmoid(ttnn.mul(x, 1.702)))

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        Returns:
            hidden_states: (batch_size, seq_len, hidden_size)

        """

        hidden_states = ttnn.linear(hidden_states, self.fc1_weight, bias=self.fc1_bias)

        hidden_states = self.quick_gelu(hidden_states)

        hidden_states = ttnn.linear(hidden_states, self.fc2_weight, bias=self.fc2_bias)

        return hidden_states


class TtCLIPAttention:
    def __init__(self, config, parameters, device):
        self.config = config
        self.device = device
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = 1.0 / (self.head_dim**0.5)

        self.q_proj_weight = ttnn.from_torch(
            parameters.q_proj.weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.q_proj_bias = ttnn.from_torch(
            parameters.q_proj.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.k_proj_weight = ttnn.from_torch(
            parameters.k_proj.weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.k_proj_bias = ttnn.from_torch(
            parameters.k_proj.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.v_proj_weight = ttnn.from_torch(
            parameters.v_proj.weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.v_proj_bias = ttnn.from_torch(
            parameters.v_proj.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.out_proj_weight = ttnn.from_torch(
            parameters.out_proj.weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.out_proj_bias = ttnn.from_torch(
            parameters.out_proj.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    def __call__(self, hidden_states: ttnn.Tensor, attention_mask: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        """
        Args:
            hidden_states: shape (batch_size, seq_len, hidden_size)
            attention_mask: shape (batch_size, seq_len) or None
        Returns:
            output: shape (batch_size, seq_len, hidden_size)
        """
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        query = ttnn.linear(hidden_states, self.q_proj_weight, bias=self.q_proj_bias)
        key = ttnn.linear(hidden_states, self.k_proj_weight, bias=self.k_proj_bias)
        value = ttnn.linear(hidden_states, self.v_proj_weight, bias=self.v_proj_bias)

        query = ttnn.reshape(query, (batch_size, seq_len, self.num_heads, self.head_dim))
        query = ttnn.permute(query, (0, 2, 1, 3))

        key = ttnn.reshape(key, (batch_size, seq_len, self.num_heads, self.head_dim))
        key = ttnn.permute(key, (0, 2, 1, 3))

        value = ttnn.reshape(value, (batch_size, seq_len, self.num_heads, self.head_dim))
        value = ttnn.permute(value, (0, 2, 1, 3))

        is_causal = seq_len > 1

        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, is_causal=is_causal, scale=self.scale
        )

        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, self.embed_dim))

        output = ttnn.linear(attn_output, self.out_proj_weight, bias=self.out_proj_bias)

        return output


class TtEncoderLayer:
    def __init__(self, config, paramters, device):
        self.config = config
        self.device = device

        self.self_attn = TtCLIPAttention(config, parameters.self_attn, device)

        self.layer_norm1_weight = ttnn.to_device(
            ttnn.from_torch(parameters.layer_norm1.weight, dtype=ttnn.bfloat16), device
        )

        self.layer_norm1_bias = ttnn.to_device(
            ttnn.from_torch(parameters.layer_norm1.bias, dtype=ttnn.bfloat16), device
        )

        self.mlp = TtCLIPMLP(config, parameters.mlp, device)

        self.layer_norm2_weight = ttnn.to_device(
            ttnn.from_torch(parameters.layer_norm2.weight, dtype=ttnn.bfloat16), device
        )

        self.layer_norm2_bias = ttnn.to_device(
            ttnn.from_torch(parameters.layer_norm2.bias, dtype=ttnn.bfloat16), device
        )

    def __call__(self, hidden_states: ttnn.Tensor, attention_mask: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        """
        Args:
            hidden_states (batch_size, seq_len, hidden_size)
            attention_mask (batch_size, seq_len)
        Returns:
            output (batch_size, seq_len, hidden_size)
        """

        residual = hidden_states
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.layer_norm1_weight,
            bias=self.layer_norm1_bias,
            epsilon=self.config.layer_norm_eps,
        )

        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = ttnn.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.layer_norm2_weight,
            bias=self.layer_norm2_bias,
            epsilon=self.config.layer_norm_eps,
        )

        hidden_states = self.mlp(hidden_states)
        hidden_states = ttnn.add(residual, hidden_states)

        return hidden_states


class TtCLIPTextModel:
    def __init__(self, reference_model, device):
        self.device = device

        self.config = reference_model.config.text_config

        self.embeddings = TtCLIPTextEmbeddings(self.config, reference_model.embeddings, device)

        self.encoder_layers = []
        for lix in range(self.config.num_hidden_layers):
            layer = TtCLIPEncoderLayer(self.config, reference_model.encoder.layers[lix], device)
            self.encoder_layers.append(layer)

        self.final_layer_norm_weight = ttnn.to_device(
            ttnn.from_torch(reference_model.final_layer_norm.weight, dtype=ttnn.bfloat16), device
        )

        self.final_layer_norm_bias = ttnn.to_device(
            ttnn.from_torch(reference_model.final_layer_norm.bias, dtype=ttnn.bfloat16), deice
        )

    def __call__(self, input_ids: ttnn.Tensor, attention_mask: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        hidden_states = self.embeddings(input_ids)

        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.final_layer_norm_weight,
            bias=self.final_layer_norm_bias,
            epsilon=self.config.layer_norm_eps,
        )

        return hidden_states

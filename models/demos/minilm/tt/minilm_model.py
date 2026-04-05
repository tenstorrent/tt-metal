# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
all-MiniLM-L6-v2: BERT encoder for sentence embeddings on Wormhole.

Architecture: 6-layer BERT encoder (384 hidden, 12 heads, 1536 intermediate)
Input: tokenized text -> Output: 384-dim sentence embedding

Uses 4D tensors [batch, 1, seq, hidden] and DRAM memory for correctness.
"""

import ttnn

# Use DRAM for all intermediate results to avoid L1 pressure
MEM = ttnn.DRAM_MEMORY_CONFIG


class MiniLMModel:
    """all-MiniLM-L6-v2 model."""

    def __init__(self, config, parameters):
        self.config = config
        self.parameters = parameters
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

    def embeddings(self, input_ids, token_type_ids, position_ids, device):
        """Word + position + token_type embeddings -> LayerNorm. Returns 4D tensor."""
        word_emb = ttnn.embedding(
            input_ids,
            self.parameters.embeddings.word_embeddings.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=MEM,
            padding_idx=self.config.pad_token_id,
        )
        token_type_emb = ttnn.embedding(
            token_type_ids,
            self.parameters.embeddings.token_type_embeddings.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=MEM,
        )
        position_emb = ttnn.embedding(
            position_ids,
            self.parameters.embeddings.position_embeddings.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=MEM,
        )

        # Unsqueeze to 4D [batch, 1, seq, hidden] before add
        word_emb = ttnn.unsqueeze(word_emb, dim=1)
        token_type_emb = ttnn.unsqueeze(token_type_emb, dim=1)
        position_emb = ttnn.unsqueeze(position_emb, dim=1)

        embeddings = ttnn.add(word_emb, token_type_emb, memory_config=MEM)
        embeddings = ttnn.add(embeddings, position_emb, memory_config=MEM)
        ttnn.deallocate(word_emb)
        ttnn.deallocate(token_type_emb)
        ttnn.deallocate(position_emb)

        # LayerNorm
        embeddings = ttnn.layer_norm(
            embeddings,
            weight=self.parameters.embeddings.LayerNorm.weight,
            bias=self.parameters.embeddings.LayerNorm.bias,
            epsilon=self.config.layer_norm_eps,
            memory_config=MEM,
        )
        return embeddings

    def attention(self, hidden_states, attention_mask, layer_params, device):
        """Multi-head self-attention. Input/output are 4D [batch, 1, seq, hidden]."""
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[2]  # 4D: [batch, 1, seq, hidden]

        # Q, K, V projections (4D in, 4D out)
        query = ttnn.linear(
            hidden_states,
            layer_params.attention.self.query.weight,
            bias=layer_params.attention.self.query.bias,
            memory_config=MEM,
        )
        key = ttnn.linear(
            hidden_states,
            layer_params.attention.self.key.weight,
            bias=layer_params.attention.self.key.bias,
            memory_config=MEM,
        )
        value = ttnn.linear(
            hidden_states,
            layer_params.attention.self.value.weight,
            bias=layer_params.attention.self.value.bias,
            memory_config=MEM,
        )

        # Squeeze to 3D for reshape to heads
        query = ttnn.squeeze(query, dim=1)
        key = ttnn.squeeze(key, dim=1)
        value = ttnn.squeeze(value, dim=1)

        # Reshape to [batch, seq, num_heads, head_dim] then permute to [batch, heads, seq, head_dim]
        query = ttnn.reshape(query, [batch_size, seq_len, self.num_heads, self.head_dim])
        key = ttnn.reshape(key, [batch_size, seq_len, self.num_heads, self.head_dim])
        value = ttnn.reshape(value, [batch_size, seq_len, self.num_heads, self.head_dim])

        query = ttnn.permute(query, (0, 2, 1, 3))
        key = ttnn.permute(key, (0, 2, 1, 3))
        value = ttnn.permute(value, (0, 2, 1, 3))

        # Attention scores: Q * K^T / sqrt(head_dim)
        key_t = ttnn.permute(key, (0, 1, 3, 2))
        ttnn.deallocate(key)

        attention_scores = ttnn.matmul(query, key_t, memory_config=MEM)
        ttnn.deallocate(query)
        ttnn.deallocate(key_t)

        attention_scores = ttnn.multiply(attention_scores, 1.0 / (self.head_dim**0.5))

        # Apply attention mask
        attention_scores = ttnn.add(attention_scores, attention_mask)

        # Softmax
        attention_probs = ttnn.softmax(attention_scores, dim=-1)
        ttnn.deallocate(attention_scores)

        # Context: softmax * V
        context = ttnn.matmul(attention_probs, value, memory_config=MEM)
        ttnn.deallocate(attention_probs)
        ttnn.deallocate(value)

        # Reshape back: [batch, heads, seq, head_dim] -> [batch, 1, seq, hidden]
        context = ttnn.permute(context, (0, 2, 1, 3))
        context = ttnn.reshape(context, [batch_size, seq_len, self.config.hidden_size])
        context = ttnn.unsqueeze(context, dim=1)

        # Output projection
        attn_output = ttnn.linear(
            context,
            layer_params.attention.output.dense.weight,
            bias=layer_params.attention.output.dense.bias,
            memory_config=MEM,
        )
        ttnn.deallocate(context)

        # Residual + LayerNorm
        attn_output = ttnn.layer_norm(
            attn_output,
            residual_input_tensor=hidden_states,
            weight=layer_params.attention.output.LayerNorm.weight,
            bias=layer_params.attention.output.LayerNorm.bias,
            epsilon=self.config.layer_norm_eps,
            memory_config=MEM,
        )
        return attn_output

    def ffn(self, hidden_states, layer_params):
        """Feed-forward network: up-project -> GELU -> down-project -> residual + LayerNorm."""
        intermediate = ttnn.linear(
            hidden_states,
            layer_params.intermediate.dense.weight,
            bias=layer_params.intermediate.dense.bias,
            memory_config=MEM,
            activation="gelu",
        )
        output = ttnn.linear(
            intermediate,
            layer_params.output.dense.weight,
            bias=layer_params.output.dense.bias,
            memory_config=MEM,
        )
        ttnn.deallocate(intermediate)

        # Residual + LayerNorm
        output = ttnn.layer_norm(
            output,
            residual_input_tensor=hidden_states,
            weight=layer_params.output.LayerNorm.weight,
            bias=layer_params.output.LayerNorm.bias,
            epsilon=self.config.layer_norm_eps,
            memory_config=MEM,
        )
        return output

    def encoder_layer(self, hidden_states, attention_mask, layer_params, device):
        """Single transformer layer: attention + FFN."""
        attn_output = self.attention(hidden_states, attention_mask, layer_params, device)
        ttnn.deallocate(hidden_states)
        output = self.ffn(attn_output, layer_params)
        return output

    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, device):
        """
        Full forward pass.

        Args:
            input_ids: [batch, seq] uint32
            token_type_ids: [batch, seq] uint32
            position_ids: [batch, seq] uint32
            attention_mask: [batch, 1, 1, seq] bfloat16 - extended attention mask
            device: TT device

        Returns:
            last_hidden_state: [batch, 1, seq, 384] (4D)
        """
        hidden_states = self.embeddings(input_ids, token_type_ids, position_ids, device)

        for i in range(self.config.num_hidden_layers):
            hidden_states = self.encoder_layer(hidden_states, attention_mask, self.parameters.encoder.layer[i], device)

        return hidden_states

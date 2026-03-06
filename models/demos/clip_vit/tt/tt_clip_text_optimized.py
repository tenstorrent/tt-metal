from typing import Optional

import torch

import ttnn


class TtCLIPTextEmbeddingsOptimized:
    def __init__(self, config, parameters, device):
        self.config = config
        self.device = device

        self.token_embedding_weight = ttnn.from_torch(
            parameters.token_embedding.weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.position_embedding_weight = ttnn.from_torch(
            parameters.position_embedding.weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )

    def __call__(self, input_ids: ttnn.Tensor, position_ids: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        """
        Args:
            input_ids: Token IDs, (batch_size, sequence_length)
            position_ids: Position IDs, (batch_size, sequence_length)
        Returns:
            embeddings: (batch_size, sequence_length, hidden_size)
        """

        seq_len = input_ids.shape[1]

        if position_ids is None:
            position_ids = ttnn.from_torch(
                torch.arange(seq_len).unsqueeze(0).expand(input_ids.shape[0], -1),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )

        token_embeddings = ttnn.embedding(input_ids, self.token_embedding_weight, layout=ttnn.TILE_LAYOUT)

        position_embeddings = ttnn.embedding(position_ids, self.position_embedding_weight, layout=ttnn.TILE_LAYOUT)

        embeddings = ttnn.add(token_embeddings, position_embeddings)

        ttnn.deallocate(token_embeddings)
        ttnn.deallocate(position_embeddings)

        return embeddings


class TtCLIPMLPOptimized:
    def __init__(self, config, parameters, device):
        self.config = config
        self.device = device

        self.fc1_weight = ttnn.from_torch(
            parameters.fc1.weight.T, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.fc1_bias = ttnn.from_torch(
            parameters.fc1.bias, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.fc2_weight = ttnn.from_torch(
            parameters.fc2.weight.T, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.fc2_bias = ttnn.from_torch(
            parameters.fc2.bias, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.core_grid = device.compute_with_storage_grid_size()

        TILE_HEIGHT = 32
        dim_t = config.hidden_size // TILE_HEIGHT
        seqL = config.max_position_embeddings
        seqL_padded = (((seqL - 1) // TILE_HEIGHT) + 1) * TILE_HEIGHT
        seqL_t = seqL_padded // TILE_HEIGHT
        dim_t_x = dim_t // self.core_grid.x

        self.fc1_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
            in0_block_w=dim_t_x,
            out_subblock_h=1,
            out_subblock_w=(dim_t_x * 4) // 2,
            per_core_M=seqL_t,
            per_core_N=dim_t_x * 4,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.fc2_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
            in0_block_w=(4 * dim_t_x),
            out_subblock_h=seqL_t,
            out_subblock_w=dim_t_x,
            per_core_M=seqL_t,
            per_core_N=dim_t_x,
            transpose_mcast=False,
            fused_activation=None,
        )

    def quick_gelu(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        x * sigmoid(x * 1.702)
        """
        return ttnn.mul(x, ttnn.sigmoid(ttnn.mul(x, 1.702)))

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        Returns:
            hidden_states: (batch_size, seq_len, hidden_size)
        """

        intermediate = ttnn.linear(
            hidden_states,
            self.fc1_weight,
            bias=self.fc1_bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            program_config=self.fc1_config,
        )

        ttnn.deallocate(hidden_states)

        intermediate = ttnn.to_memory_config(intermediate, ttnn.L1_MEMORY_CONFIG)
        intermediate = self.quick_gelu(intermediate)

        output = ttnn.linear(
            intermediate,
            self.fc2_weight,
            bias=self.fc2_bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            program_config=self.fc2_config,
        )

        ttnn.deallocate(intermediate)

        return output


class TtCLIPAttentionOptimized:
    def __init__(self, config, parameters, device):
        self.config = config
        self.device = device
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = 1.0 / (self.head_dim**0.5)

        qkv_weight = torch.cat(
            [parameters.q_proj.weight, parameters.k_proj.weight, parameters.v_proj.weight],
            dim=0,
        )

        qkv_bias = torch.cat(
            [parameters.q_proj.bias, parameters.k_proj.bias, parameters.v_proj.bias],
            dim=0,
        ).reshape(1, 3 * self.embed_dim)

        self.qkv_weight = ttnn.from_torch(qkv_weight.T, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

        self.qkv_bias = ttnn.from_torch(qkv_bias, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

        self.out_proj_weight = ttnn.from_torch(
            parameters.out_proj.weight.T, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.out_proj_bias = ttnn.from_torch(
            parameters.out_proj.bias.reshape(1, -1), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )

        TILE_HEIGHT = 32
        self.core_grid = device.compute_with_storage_grid_size()
        dim_t = config.hidden_size // TILE_HEIGHT
        seqL = config.max_position_embeddings
        seqL_padded = (((seqL - 1) // TILE_HEIGHT) + 1) * TILE_HEIGHT
        seqL_t = seqL_padded // TILE_HEIGHT
        dim_t_x = dim_t // self.core_grid.x

        self.qkv_linear_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
            in0_block_w=dim_t_x,
            out_subblock_h=1,
            out_subblock_w=dim_t_x,
            per_core_M=seqL_t,
            per_core_N=(3 * dim_t_x),
            transpose_mcast=False,
            fused_activation=None,
        )

        self.sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
            exp_approx_mode=False,
            q_chunk_size=64,
            k_chunk_size=64,
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        self.out_proj_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
            in0_block_w=dim_t_x,
            out_subblock_h=seqL_t,
            out_subblock_w=dim_t_x,
            per_core_M=seqL_t,
            per_core_N=dim_t_x,
            transpose_mcast=False,
            fused_activation=None,
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

        qkv = ttnn.linear(
            hidden_states,
            self.qkv_weight,
            bias=self.qkv_bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            program_config=self.qkv_linear_config,
        )

        ttnn.deallocate(hidden_states)

        qkv = ttnn.to_memory_config(qkv, ttnn.L1_MEMORY_CONFIG)

        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=self.num_heads, memory_config=ttnn.L1_MEMORY_CONFIG, transpose_key=False
        )

        ttnn.deallocate(qkv)

        is_causal = seq_len > 1

        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            is_causal=is_causal,
            scale=self.scale,
            program_config=self.sdpa_config,
            compute_kernel_config=self.compute_kernel_config,
        )

        ttnn.deallocate(query)
        ttnn.deallocate(key)
        ttnn.deallocate(value)

        attn_output = ttnn.transformer.concatenate_heads(
            attn_output,
        )

        output = ttnn.linear(
            attn_output,
            self.out_proj_weight,
            bias=self.out_proj_bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            program_config=self.out_proj_config,
        )

        ttnn.deallocate(attn_output)

        return output


class TtCLIPEncoderLayerOptimized:
    def __init__(self, config, parameters, device):
        self.config = config
        self.device = device

        self.self_attn = TtCLIPAttentionOptimized(config, parameters.self_attn, device)

        self.layer_norm1_weight = ttnn.from_torch(
            parameters.layer_norm1.weight.reshape(1, self.config.hidden_size // 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        self.layer_norm1_bias = ttnn.from_torch(
            parameters.layer_norm1.bias.reshape(1, self.config.hidden_size // 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        self.mlp = TtCLIPMLPOptimized(config, parameters.mlp, device)

        self.layer_norm2_weight = ttnn.from_torch(
            parameters.layer_norm2.weight.reshape(1, self.config.hidden_size // 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        self.layer_norm2_bias = ttnn.from_torch(
            parameters.layer_norm2.bias.reshape(1, self.config.hidden_size // 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
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
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        hidden_states = self.self_attn(hidden_states, attention_mask)

        hidden_states = ttnn.add(
            residual,
            hidden_states,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        ttnn.deallocate(residual)

        residual = hidden_states
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.layer_norm2_weight,
            bias=self.layer_norm2_bias,
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        hidden_states = self.mlp(hidden_states)

        hidden_states = ttnn.add(
            residual,
            hidden_states,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        ttnn.deallocate(residual)

        return hidden_states


class TtCLIPTextModelOptimized:
    def __init__(self, config, parameters, device):
        self.device = device
        self.config = config

        self.embeddings = TtCLIPTextEmbeddingsOptimized(self.config, parameters.embeddings, device)

        self.encoder_layers = []
        for lix in range(self.config.num_hidden_layers):
            layer = TtCLIPEncoderLayerOptimized(self.config, parameters.encoder.layers[lix], device)
            self.encoder_layers.append(layer)

        self.final_layer_norm_weight = ttnn.from_torch(
            parameters.final_layer_norm.weight.reshape(1, self.config.hidden_size // 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        self.final_layer_norm_bias = ttnn.from_torch(
            parameters.final_layer_norm.bias.reshape(1, self.config.hidden_size // 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

    def __call__(
        self,
        input_ids: ttnn.Tensor,
        position_ids: Optional[ttnn.Tensor] = None,
        attention_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        hidden_states = self.embeddings(input_ids, position_ids)

        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.final_layer_norm_weight,
            bias=self.final_layer_norm_bias,
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        return hidden_states

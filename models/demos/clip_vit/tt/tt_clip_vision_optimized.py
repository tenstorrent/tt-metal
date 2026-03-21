import torch

import ttnn


class TtCLIPVisionEmbeddings:
    def __init__(self, config, parameters, device):
        self.config = config
        self.device = device
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        core_grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=core_grid.y, x=core_grid.x)

        patch_weight = parameters.patch_embedding.weight  # (768, 3, 32, 32)
        out_channels, c, kh, kw = patch_weight.shape

        patch_weight = torch.nn.functional.pad(patch_weight, (0, 0, 0, 0, 0, 4 - c))
        patch_weight = torch.permute(patch_weight, (2, 3, 1, 0))
        patch_weight = torch.reshape(patch_weight, (kh * kw * 4, out_channels))  # (4096, 768)

        self.patch_embedding_weight = ttnn.from_torch(
            patch_weight,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.position_ids = ttnn.from_torch(
            torch.arange(self.num_positions).unsqueeze(0),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

        self.position_embedding_weight = ttnn.from_torch(
            parameters.position_embedding.weight,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.class_embedding = ttnn.from_torch(
            parameters.class_embedding.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.position_embeddings = ttnn.embedding(
            self.position_ids, self.position_embedding_weight, layout=ttnn.TILE_LAYOUT
        )

    def __call__(self, pixel_values: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            pixel_values: Input images, shape (batch_size, num_channels, height, width)
        Returns:
            embeddings: shape (batch_size, num_patches + 1, hidden_size)
        """
        batch_size, n_channels, height, width = pixel_values.shape

        pixel_values = self._preprocess_pixels(pixel_values)

        patch_embeds = ttnn.linear(
            pixel_values,
            self.patch_embedding_weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=self.core_grid,
        )

        ttnn.deallocate(pixel_values)

        patch_embeds = ttnn.to_layout(patch_embeds, layout=ttnn.ROW_MAJOR_LAYOUT)
        patch_embeds = ttnn.reshape(patch_embeds, (batch_size, self.num_patches, self.embed_dim))
        patch_embeds = ttnn.to_layout(patch_embeds, layout=ttnn.TILE_LAYOUT)

        class_embeds = ttnn.repeat(self.class_embedding, ttnn.Shape([batch_size, 1, 1]))

        embeddings = ttnn.concat([class_embeds, patch_embeds], dim=1)

        ttnn.deallocate(patch_embeds)

        embeddings = ttnn.add(embeddings, self.position_embeddings)

        return embeddings

    def _preprocess_pixels(self, pixel_values):
        """
        preprocess input pixels so we can do a standard matmul equivalent to a conv2d
            1) permute [n,c=3,h=224,w=224] -> [n,224,224,3]
            2) pad to channel [n,224,224,3] -> [n,224,224,4]
            3) reshape [n,224,224,4] -> [n,224,224//32, 4*32] = [n,224,7,128]
            4) fold [n,224,7,128] -> [n*49, 128*(224//7)]  = [n*49, 4096]
        """

        batch_size, n_channels, height, width = pixel_values.shape
        pixel_values = ttnn.permute(pixel_values, [0, 2, 3, 1])
        pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.ROW_MAJOR_LAYOUT)
        pixel_values = ttnn.pad(pixel_values, padding=((0, 0), (0, 0), (0, 0), (0, 1)), value=0)
        pixel_values = ttnn.reshape(pixel_values, (batch_size, height, width // self.patch_size, 4 * self.patch_size))
        pixel_values = ttnn.fold(pixel_values, stride_h=self.patch_size, stride_w=1)
        pixel_values = ttnn.reallocate(pixel_values)
        pixel_values = ttnn.to_memory_config(pixel_values, memory_config=ttnn.L1_MEMORY_CONFIG)
        pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

        return pixel_values


class TtCLIPVisionMLP:
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
        patch_count = config.image_size // config.patch_size
        seqL = patch_count * patch_count
        seqL_padded = (((seqL - 1) // TILE_HEIGHT) + 1) * TILE_HEIGHT
        seqL_t = seqL_padded // TILE_HEIGHT
        dim_t_x = dim_t // self.core_grid.x
        head_num = config.num_attention_heads

        self.fc1_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
            in0_block_w=dim_t_x,
            out_subblock_h=1,
            out_subblock_w=(dim_t_x * 4) // 2,
            per_core_M=seqL_t,
            per_core_N=dim_t_x * 4,
            transpose_mcast=False,
            fused_activation=(ttnn.UnaryOpType.GELU, True),
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


class TtCLIPVisionAttention:
    def __init__(self, config, parameters, device):
        self.config = config
        self.device = device
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = 1.0 / (self.head_dim**0.5)

        num_heads = self.num_heads
        head_dim = self.head_dim
        qkv_weight = torch.cat(
            [
                parameters.q_proj.weight.reshape([num_heads, head_dim, -1]),
                parameters.k_proj.weight.reshape([num_heads, head_dim, -1]),
                parameters.v_proj.weight.reshape([num_heads, head_dim, -1]),
            ],
            dim=1,
        ).reshape(3 * self.embed_dim, -1)

        qkv_bias = torch.cat(
            [
                parameters.q_proj.bias.reshape([num_heads, head_dim]),
                parameters.k_proj.bias.reshape([num_heads, head_dim]),
                parameters.v_proj.bias.reshape([num_heads, head_dim]),
            ],
            dim=1,
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
        patch_count = config.image_size // config.patch_size
        seqL = patch_count * patch_count
        seqL_padded = (((seqL - 1) // TILE_HEIGHT) + 1) * TILE_HEIGHT
        seqL_t = seqL_padded // TILE_HEIGHT
        dim_t_x = dim_t // self.core_grid.x
        head_num = config.num_attention_heads

        self.qkv_linear_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
            in0_block_w=dim_t_x,
            out_subblock_h=1,
            out_subblock_w=dim_t_x,
            per_core_M=seqL_t,
            per_core_N=(3 * dim_t),
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

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            hidden_states: shape (batch_size, seq_len, hidden_size)
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

        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=self.num_heads, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG, transpose_key=False
        )

        ttnn.deallocate(qkv)

        query = ttnn.to_memory_config(query, ttnn.L1_MEMORY_CONFIG)
        key = ttnn.to_memory_config(key, ttnn.L1_MEMORY_CONFIG)
        value = ttnn.to_memory_config(value, ttnn.L1_MEMORY_CONFIG)

        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            is_causal=False,
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


class TtCLIPVisionEncoderLayer:
    def __init__(self, config, parameters, device):
        self.config = config
        self.device = device

        self.self_attn = TtCLIPVisionAttention(config, parameters.self_attn, device)

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

        self.mlp = TtCLIPVisionMLP(config, parameters.mlp, device)

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

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        Returns:
            output: (batch_size, seq_len, hidden_size)
        """
        residual = hidden_states
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.layer_norm1_weight,
            bias=self.layer_norm1_bias,
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        hidden_states = self.self_attn(hidden_states)

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


class TtCLIPVisionModel:
    def __init__(self, config, parameters, device):
        self.device = device
        self.config = config

        self.embeddings = TtCLIPVisionEmbeddings(self.config, parameters.embeddings, device)

        self.pre_layernorm_weight = ttnn.from_torch(
            parameters.pre_layrnorm.weight.reshape(1, self.config.hidden_size // 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        self.pre_layernorm_bias = ttnn.from_torch(
            parameters.pre_layrnorm.bias.reshape(1, self.config.hidden_size // 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        self.encoder_layers = []
        for lix in range(self.config.num_hidden_layers):
            layer = TtCLIPVisionEncoderLayer(self.config, parameters.encoder.layers[lix], device)
            self.encoder_layers.append(layer)

        self.post_layernorm_weight = ttnn.from_torch(
            parameters.post_layernorm.weight.reshape(1, self.config.hidden_size // 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        self.post_layernorm_bias = ttnn.from_torch(
            parameters.post_layernorm.bias.reshape(1, self.config.hidden_size // 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

    def __call__(
        self,
        pixel_values: ttnn.Tensor,
    ):
        """
        Args:
            pixel_values: Input images, shape (batch_size, num_channels, height, width)
        Returns:
            pooler_output: shape (batch_size, hidden_size) - the CLS token output after post_layernorm
        """

        hidden_states = self.embeddings(pixel_values)

        ttnn.deallocate(pixel_values)

        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.pre_layernorm_weight,
            bias=self.pre_layernorm_bias,
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states)

        batch_size = hidden_states.shape[0]
        hidden_size = hidden_states.shape[2]

        pooled_output = ttnn.slice(hidden_states, [0, 0, 0], [batch_size, 1, hidden_size])

        ttnn.deallocate(hidden_states)

        pooled_output = ttnn.reshape(pooled_output, (batch_size, hidden_size))

        pooled_output = ttnn.layer_norm(
            pooled_output,
            weight=self.post_layernorm_weight,
            bias=self.post_layernorm_bias,
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        return pooled_output

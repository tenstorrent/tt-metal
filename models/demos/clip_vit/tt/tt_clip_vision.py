from typing import Optional, Tuple

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

        # Class embedding (CLS token) - shape: (embed_dim,) -> (1, 1, embed_dim)
        self.class_embedding = ttnn.from_torch(
            parameters.class_embedding.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # Patch embedding is Conv2d with kernel_size=patch_size, stride=patch_size
        # HuggingFace shape: (embed_dim, num_channels, patch_size, patch_size)
        self.patch_embedding_weight = ttnn.from_torch(
            parameters.patch_embedding.weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        # Position embedding - shape: (num_positions, embed_dim)
        self.position_embedding_weight = ttnn.from_torch(
            parameters.position_embedding.weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def __call__(self, pixel_values: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            pixel_values: Input images, shape (batch_size, num_channels, height, width)
        Returns:
            embeddings: shape (batch_size, num_patches + 1, hidden_size)
        """
        batch_size = pixel_values.shape[0]

        patch_embeds = ttnn.conv2d(
            input_tensor=pixel_values,
            weight_tensor=self.patch_embedding_weight,
            device=self.device,
            in_channels=self.num_channels,
            out_channels=self.embed_dim,
            batch_size=batch_size,
            input_height=self.image_size,
            input_width=self.image_size,
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
        )

        patch_embeds = ttnn.reshape(patch_embeds, (batch_size, self.num_patches, self.embed_dim))

        # Expand class embedding to batch size: (1, 1, embed_dim) -> (batch, 1, embed_dim)
        class_embeds = ttnn.repeat(self.class_embedding, ttnn.Shape([batch_size, 1, 1]))

        # Concatenate [CLS, patch_1, patch_2, ..., patch_n]
        embeddings = ttnn.concat([class_embeds, patch_embeds], dim=1)

        # Add position embeddings
        position_ids = ttnn.from_torch(
            torch.arange(self.num_positions).unsqueeze(0).expand(batch_size, -1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        position_embeddings = ttnn.embedding(position_ids, self.position_embedding_weight, layout=ttnn.TILE_LAYOUT)

        embeddings = ttnn.add(embeddings, position_embeddings)

        return embeddings


class TtCLIPVisionMLP:
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
        hidden_states = ttnn.linear(hidden_states, self.fc1_weight, bias=self.fc1_bias)

        hidden_states = self.quick_gelu(hidden_states)

        hidden_states = ttnn.linear(hidden_states, self.fc2_weight, bias=self.fc2_bias)

        return hidden_states


class TtCLIPVisionAttention:
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

        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, is_causal=False, scale=self.scale
        )

        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, self.embed_dim))

        output = ttnn.linear(attn_output, self.out_proj_weight, bias=self.out_proj_bias)

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

    def __call__(self, hidden_states: ttnn.Tensor, attention_mask: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len) or None
        Returns:
            output: (batch_size, seq_len, hidden_size)
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
        attention_mask: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Args:
            pixel_values: Input images, shape (batch_size, num_channels, height, width)
            attention_mask: Optional attention mask
        Returns:
            last_hidden_state: shape (batch_size, num_patches + 1, hidden_size)
            pooler_output: shape (batch_size, hidden_size) - the CLS token output after post_layernorm
        """
        hidden_states = self.embeddings(pixel_values)

        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.pre_layernorm_weight,
            bias=self.pre_layernorm_bias,
            epsilon=self.config.layer_norm_eps,
        )

        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask)

        last_hidden_state = hidden_states

        batch_size = hidden_states.shape[0]
        hidden_size = hidden_states.shape[2]

        pooled_output = ttnn.slice(hidden_states, [0, 0, 0], [batch_size, 1, hidden_size])

        pooled_output = ttnn.reshape(pooled_output, (batch_size, hidden_size))

        pooled_output = ttnn.layer_norm(
            pooled_output,
            weight=self.post_layernorm_weight,
            bias=self.post_layernorm_bias,
            epsilon=self.config.layer_norm_eps,
        )

        return last_hidden_state, pooled_output

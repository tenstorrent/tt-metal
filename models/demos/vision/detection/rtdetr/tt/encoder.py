import torch

import ttnn
from models.demos.vision.detection.rtdetr.tt.backbone import TtRTDetrResNetConvLayer


def build_2d_sinusoidal_position_embedding(
    height: int,
    width: int,
    embed_dim: int,
    temperature: float,
) -> torch.Tensor:
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")

    position_dim = embed_dim // 4
    omega = torch.arange(position_dim, dtype=torch.float64) / position_dim
    omega = 1.0 / temperature**omega

    grid_h = torch.arange(height, dtype=torch.float64)
    grid_w = torch.arange(width, dtype=torch.float64)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")

    embedding_h = grid_h.flatten().outer(omega)
    embedding_w = grid_w.flatten().outer(omega)

    position_embedding = torch.cat(
        [embedding_h.sin(), embedding_h.cos(), embedding_w.sin(), embedding_w.cos()],
        dim=-1,
    )
    return position_embedding.to(torch.float32).unsqueeze(0)


class TtRTDetrConvNormLayer(TtRTDetrResNetConvLayer):
    def __init__(
        self,
        config,
        parameters,
        device,
        dtype,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        activation: str,
    ):
        if activation not in ("silu", "identity"):
            raise ValueError(f"Unsupported ConvNorm activation: {activation}")

        super().__init__(
            config=config,
            parameters=parameters,
            device=device,
            dtype=dtype,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation=activation,
        )


class TtRTDetrHybridEncoder:
    def __init__(self, config, parameters, device, dtype):
        self.in_channels = config.encoder_in_channels
        self.feat_strides = config.feat_strides
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.encoder_layers = config.encoder_layers
        self.encode_proj_layers = config.encode_proj_layers
        self.positional_encoding_temperature = config.positional_encoding_temperature
        self.eval_size = config.eval_size
        self.out_channels = [self.encoder_hidden_dim for _ in self.in_channels]
        self.out_strides = self.feat_strides
        self.num_fpn_stages = len(self.in_channels) - 1
        self.num_pan_stages = len(self.in_channels) - 1

        self.aifi = [
            TtRTDetrAIFILayer(
                config=config,
                parameters=parameters.aifi[ix],
                device=device,
                dtype=dtype,
            )
            for ix in range(len(self.encode_proj_layers))
        ]

        # top-down FPN
        self.lateral_convs = []
        self.fpn_blocks = []
        for ix in range(self.num_fpn_stages):
            lateral_conv = TtRTDetrConvNormLayer(
                config=config,
                parameters=parameters.lateral_convs[ix],
                device=device,
                dtype=dtype,
                in_channels=self.encoder_hidden_dim,
                out_channels=self.encoder_hidden_dim,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                activation=config.activation_function,
            )
            fpn_block = TtRTDetrCSPRepLayer(
                config=config,
                parameters=parameters.fpn_blocks[ix],
                device=device,
                dtype=dtype,
            )
            self.lateral_convs.append(lateral_conv)
            self.fpn_blocks.append(fpn_block)

        # bottom-up PAN
        self.downsample_convs = []
        self.pan_blocks = []
        for ix in range(self.num_pan_stages):
            downsample_conv = TtRTDetrConvNormLayer(
                config=config,
                parameters=parameters.downsample_convs[ix],
                device=device,
                dtype=dtype,
                in_channels=self.encoder_hidden_dim,
                out_channels=self.encoder_hidden_dim,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                activation=config.activation_function,
            )
            pan_block = TtRTDetrCSPRepLayer(
                config=config,
                parameters=parameters.pan_blocks[ix],
                device=device,
                dtype=dtype,
            )
            self.downsample_convs.append(downsample_conv)
            self.pan_blocks.append(pan_block)

    @staticmethod
    def _to_dram_row_major(hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)

    def _upsample(
        self,
        hidden_states: ttnn.Tensor,
        batch_size: int,
        height: int,
        width: int,
    ) -> tuple[ttnn.Tensor, int, int]:
        hidden_states = self._to_dram_row_major(hidden_states)
        hidden_states = ttnn.reshape(
            hidden_states,
            (batch_size, height, width, self.encoder_hidden_dim),
        )
        hidden_states = ttnn.upsample(
            hidden_states,
            scale_factor=2,
            mode="nearest",
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        height *= 2
        width *= 2
        hidden_states = ttnn.reshape(
            hidden_states,
            (1, 1, batch_size * height * width, self.encoder_hidden_dim),
        )
        return hidden_states, height, width

    def _concat(self, hidden_states_1: ttnn.Tensor, hidden_states_2: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.concat(
            [
                self._to_dram_row_major(hidden_states_1),
                self._to_dram_row_major(hidden_states_2),
            ],
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(
        self,
        inputs_embeds: list[tuple[ttnn.Tensor, int, int]],
        batch_size: int,
    ) -> list[tuple[ttnn.Tensor, int, int]]:
        feature_maps = list(inputs_embeds)

        if self.encoder_layers > 0:
            for ix, feature_index in enumerate(self.encode_proj_layers):
                hidden_states, height, width = feature_maps[feature_index]
                feature_maps[feature_index] = self.aifi[ix](
                    hidden_states,
                    batch_size=batch_size,
                    height=height,
                    width=width,
                )

        fpn_feature_maps = [feature_maps[-1]]
        for ix, (lateral_conv, fpn_block) in enumerate(zip(self.lateral_convs, self.fpn_blocks)):
            backbone_feature_map, _, _ = feature_maps[self.num_fpn_stages - ix - 1]
            top_feature_map, top_height, top_width = fpn_feature_maps[-1]

            top_feature_map, top_height, top_width = lateral_conv(
                top_feature_map,
                batch_size=batch_size,
                input_height=top_height,
                input_width=top_width,
            )
            fpn_feature_maps[-1] = (top_feature_map, top_height, top_width)

            top_feature_map, top_height, top_width = self._upsample(
                top_feature_map,
                batch_size=batch_size,
                height=top_height,
                width=top_width,
            )
            fused_feature_map = self._concat(top_feature_map, backbone_feature_map)
            new_fpn_feature_map = fpn_block(
                fused_feature_map,
                batch_size=batch_size,
                input_height=top_height,
                input_width=top_width,
            )
            fpn_feature_maps.append(new_fpn_feature_map)

        fpn_feature_maps.reverse()

        pan_feature_maps = [fpn_feature_maps[0]]
        for ix, (downsample_conv, pan_block) in enumerate(zip(self.downsample_convs, self.pan_blocks)):
            top_feature_map, top_height, top_width = pan_feature_maps[-1]
            fpn_feature_map, _, _ = fpn_feature_maps[ix + 1]

            downsampled_feature_map, downsampled_height, downsampled_width = downsample_conv(
                top_feature_map,
                batch_size=batch_size,
                input_height=top_height,
                input_width=top_width,
            )
            fused_feature_map = self._concat(downsampled_feature_map, fpn_feature_map)
            new_pan_feature_map = pan_block(
                fused_feature_map,
                batch_size=batch_size,
                input_height=downsampled_height,
                input_width=downsampled_width,
            )
            pan_feature_maps.append(new_pan_feature_map)

        return pan_feature_maps


class TtRTDetrRepVggBlock:
    def __init__(self, config, parameters, device, dtype):
        hidden_channels = int(config.encoder_hidden_dim * config.hidden_expansion)
        self.activation = config.activation_function

        self.conv1 = TtRTDetrConvNormLayer(
            config=config,
            parameters=parameters.conv1,
            device=device,
            dtype=dtype,
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            activation="identity",
        )
        self.conv2 = TtRTDetrConvNormLayer(
            config=config,
            parameters=parameters.conv2,
            device=device,
            dtype=dtype,
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation="identity",
        )

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        batch_size: int,
        input_height: int,
        input_width: int,
    ) -> tuple[ttnn.Tensor, int, int]:
        hidden_states_1, output_height, output_width = self.conv1(hidden_states, batch_size, input_height, input_width)
        hidden_states_2, _, _ = self.conv2(hidden_states, batch_size, input_height, input_width)
        hidden_states = ttnn.add(hidden_states_1, hidden_states_2)

        if self.activation == "silu":
            hidden_states = ttnn.silu(hidden_states)
        elif self.activation is not None:
            raise ValueError(f"Unsupported RepVGG activation: {self.activation}")

        return hidden_states, output_height, output_width


class TtRTDetrCSPRepLayer:
    def __init__(self, config, parameters, device, dtype):
        in_channels = config.encoder_hidden_dim * 2
        out_channels = config.encoder_hidden_dim
        hidden_channels = int(out_channels * config.hidden_expansion)

        self.conv1 = TtRTDetrConvNormLayer(
            config=config,
            parameters=parameters.conv1,
            device=device,
            dtype=dtype,
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation=config.activation_function,
        )
        self.conv2 = TtRTDetrConvNormLayer(
            config=config,
            parameters=parameters.conv2,
            device=device,
            dtype=dtype,
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation=config.activation_function,
        )
        self.bottlenecks = [
            TtRTDetrRepVggBlock(
                config=config,
                parameters=parameters.bottlenecks[ix],
                device=device,
                dtype=dtype,
            )
            for ix in range(3)
        ]
        self.conv3 = None
        if hidden_channels != out_channels:
            self.conv3 = TtRTDetrConvNormLayer(
                config=config,
                parameters=parameters.conv3,
                device=device,
                dtype=dtype,
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                activation=config.activation_function,
            )

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        batch_size: int,
        input_height: int,
        input_width: int,
    ) -> tuple[ttnn.Tensor, int, int]:
        hidden_states_1, output_height, output_width = self.conv1(hidden_states, batch_size, input_height, input_width)

        for bottleneck in self.bottlenecks:
            hidden_states_1, output_height, output_width = bottleneck(
                hidden_states_1, batch_size, output_height, output_width
            )

        hidden_states_2, _, _ = self.conv2(hidden_states, batch_size, input_height, input_width)

        hidden_states = ttnn.add(hidden_states_1, hidden_states_2)

        if self.conv3 is not None:
            hidden_states, output_height, output_width = self.conv3(
                hidden_states, batch_size, output_height, output_width
            )

        return hidden_states, output_height, output_width


class TtRTDetrAIFILayer:
    def __init__(self, config, parameters, device, dtype):
        self.device = device
        self.dtype = dtype
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.positional_encoding_temp = config.positional_encoding_temperature
        self.position_embedding_cache = {}

        self.layers = [
            TtRTDetrEncoderLayer(config, parameters=parameters.layers[ix], device=device, dtype=dtype)
            for ix in range(config.encoder_layers)
        ]

    def _build_sinusoidal_position_embedding(self, height: int, width: int) -> ttnn.Tensor:
        cache_key = (height, width)
        if cache_key not in self.position_embedding_cache:
            position_embedding = build_2d_sinusoidal_position_embedding(
                height=height,
                width=width,
                embed_dim=self.encoder_hidden_dim,
                temperature=self.positional_encoding_temp,
            )
            self.position_embedding_cache[cache_key] = ttnn.from_torch(
                position_embedding,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

        return self.position_embedding_cache[cache_key]

    def __call__(
        self, hidden_states: ttnn.Tensor, batch_size: int, height: int, width: int
    ) -> tuple[ttnn.Tensor, int, int]:
        hidden_states = ttnn.to_memory_config(
            hidden_states,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden_states = ttnn.to_layout(
            hidden_states,
            ttnn.TILE_LAYOUT,
        )
        hidden_states = ttnn.reshape(
            hidden_states,
            (batch_size, height * width, self.encoder_hidden_dim),
        )

        position_embeddings = self._build_sinusoidal_position_embedding(height, width)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings)

        hidden_states = ttnn.reshape(
            hidden_states,
            (1, 1, batch_size * height * width, self.encoder_hidden_dim),
        )

        return hidden_states, height, width


class TtRTDetrEncoderLayer:
    def __init__(self, config, parameters, device, dtype):
        self.self_attn = TtRTDetrSelfAttention(config, parameters.self_attn, device, dtype)

        self.mlp = TtRTDetrMLP(config, parameters.mlp, device, dtype)

        self.layer_norm_eps = config.layer_norm_eps

        self.self_attn_layer_norm_weight = ttnn.from_torch(
            parameters.self_attn_layer_norm.weight, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.self_attn_layer_norm_bias = ttnn.from_torch(
            parameters.self_attn_layer_norm.bias, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.final_layer_norm_weight = ttnn.from_torch(
            parameters.final_layer_norm.weight, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.final_layer_norm_bias = ttnn.from_torch(
            parameters.final_layer_norm.bias, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

    def __call__(self, hidden_states: ttnn.Tensor, position_embeddings: ttnn.Tensor = None) -> ttnn.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, position_embeddings=position_embeddings)
        hidden_states = ttnn.layer_norm(
            hidden_states,
            epsilon=self.layer_norm_eps,
            weight=self.self_attn_layer_norm_weight,
            bias=self.self_attn_layer_norm_bias,
            residual_input_tensor=residual,
        )

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = ttnn.layer_norm(
            hidden_states,
            epsilon=self.layer_norm_eps,
            weight=self.final_layer_norm_weight,
            bias=self.final_layer_norm_bias,
            residual_input_tensor=residual,
        )

        return hidden_states


class TtRTDetrSelfAttention:
    def __init__(self, config, parameters, device, dtype):
        self.embed_dim = config.encoder_hidden_dim
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj_weight = ttnn.from_torch(
            parameters.q_proj.weight.T, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.q_proj_bias = ttnn.from_torch(parameters.q_proj.bias, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        self.k_proj_weight = ttnn.from_torch(
            parameters.k_proj.weight.T, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.k_proj_bias = ttnn.from_torch(parameters.k_proj.bias, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        self.v_proj_weight = ttnn.from_torch(
            parameters.v_proj.weight.T, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.v_proj_bias = ttnn.from_torch(parameters.v_proj.bias, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        self.o_proj_weight = ttnn.from_torch(
            parameters.o_proj.weight.T, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.o_proj_bias = ttnn.from_torch(parameters.o_proj.bias, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    def __call__(self, hidden_states: ttnn.Tensor, position_embeddings: ttnn.Tensor = None) -> ttnn.Tensor:
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        qk_input = hidden_states
        if position_embeddings is not None:
            qk_input = ttnn.add(hidden_states, position_embeddings)

        query = ttnn.linear(qk_input, self.q_proj_weight, bias=self.q_proj_bias)
        key = ttnn.linear(qk_input, self.k_proj_weight, bias=self.k_proj_bias)
        value = ttnn.linear(hidden_states, self.v_proj_weight, bias=self.v_proj_bias)

        query = ttnn.reshape(query, (batch_size, seq_len, self.num_heads, self.head_dim))
        query = ttnn.permute(query, (0, 2, 1, 3))

        key = ttnn.reshape(key, (batch_size, seq_len, self.num_heads, self.head_dim))
        key = ttnn.permute(key, (0, 2, 1, 3))

        value = ttnn.reshape(value, (batch_size, seq_len, self.num_heads, self.head_dim))
        value = ttnn.permute(value, (0, 2, 1, 3))

        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query, key, value, attn_mask=None, is_causal=False, scale=self.scale
        )

        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, self.embed_dim))

        output = ttnn.linear(attn_output, self.o_proj_weight, bias=self.o_proj_bias)

        return output


class TtRTDetrMLP:
    def __init__(self, config, parameters, device, dtype):
        self.fc1_weight = ttnn.from_torch(parameters.fc1.weight.T, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.fc1_bias = ttnn.from_torch(parameters.fc1.bias, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.fc2_weight = ttnn.from_torch(parameters.fc2.weight.T, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.fc2_bias = ttnn.from_torch(parameters.fc2.bias, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        hidden_states = ttnn.linear(hidden_states, self.fc1_weight, bias=self.fc1_bias)
        hidden_states = ttnn.gelu(hidden_states)
        hidden_states = ttnn.linear(hidden_states, self.fc2_weight, bias=self.fc2_bias)

        return hidden_states

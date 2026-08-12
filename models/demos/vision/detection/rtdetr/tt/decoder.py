# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.vision.detection.rtdetr.tt.encoder import TtRTDetrSelfAttention


class TtRTDetrMLPPredictionHead:
    def __init__(
        self,
        config,
        parameters,
        device,
        dtype,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.weights = []
        self.biases = []

        for index in range(num_layers):
            self.weights.append(
                ttnn.from_torch(
                    parameters.layers[index].weight.T,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
            )
            self.biases.append(
                ttnn.from_torch(
                    parameters.layers[index].bias,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
            )

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            hidden_states = ttnn.linear(
                hidden_states,
                weight,
                bias=bias,
                activation="relu",
            )

        hidden_states = ttnn.linear(
            hidden_states,
            self.weights[-1],
            bias=self.biases[-1],
        )

        return hidden_states


class TtRTDetrDecoderMLP:
    def __init__(self, config, parameters, device, dtype):
        self.fc1_weight = ttnn.from_torch(
            parameters.fc1.weight.T,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.fc1_bias = ttnn.from_torch(
            parameters.fc1.bias,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.fc2_weight = ttnn.from_torch(
            parameters.fc2.weight.T,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.fc2_bias = ttnn.from_torch(
            parameters.fc2.bias,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.activation = config.decoder_activation_function

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        hidden_states = ttnn.linear(
            hidden_states,
            self.fc1_weight,
            bias=self.fc1_bias,
            activation=self.activation,
        )
        hidden_states = ttnn.linear(
            hidden_states,
            self.fc2_weight,
            bias=self.fc2_bias,
        )

        return hidden_states


class TtRTDetrDecoderSelfAttention(TtRTDetrSelfAttention):
    def __init__(self, config, parameters, device, dtype):
        super().__init__(
            config=config,
            parameters=parameters,
            device=device,
            dtype=dtype,
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
        )


class TtRTDetrMultiscaleDeformableAttention:
    def __init__(self, config, parameters, device, dtype):
        self.num_heads = config.decoder_attention_heads
        self.n_points = config.decoder_n_points
        self.d_model = config.d_model
        self.n_levels = getattr(config, "decoder_n_levels", config.num_feature_levels)
        self.offset_scale = getattr(config, "decoder_offset_scale", 0.5)

        self.im2col_step = 64

        self.sampling_offsets_weight = ttnn.from_torch(
            parameters.sampling_offsets.weight.T,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.sampling_offsets_bias = ttnn.from_torch(
            parameters.sampling_offsets.bias,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.attention_weights_weight = ttnn.from_torch(
            parameters.attention_weights.weight.T,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.attention_weights_bias = ttnn.from_torch(
            parameters.attention_weights.bias,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.value_proj_weight = ttnn.from_torch(
            parameters.value_proj.weight.T,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.value_proj_bias = ttnn.from_torch(
            parameters.value_proj.bias,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.output_proj_weight = ttnn.from_torch(
            parameters.output_proj.weight.T,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.output_proj_bias = ttnn.from_torch(
            parameters.output_proj.bias,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    @staticmethod
    def _deformable_attention(
        value: ttnn.Tensor,
        value_spatial_shapes: ttnn.Tensor,
        value_spatial_shapes_list: tuple[tuple[int, int], ...],
        level_start_index: ttnn.Tensor,
        sampling_locations: ttnn.Tensor,
        attention_weights: ttnn.Tensor,
        im2col_step: int,
    ):
        batch_size, sequence_length, num_heads, hidden_dim = value.shape
        _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape

        level_sizes = [height * width for height, width in value_spatial_shapes_list]

        value = ttnn.permute(value, (0, 2, 1, 3))
        value = ttnn.reshape(value, (batch_size * num_heads, sequence_length, hidden_dim))
        value = ttnn.to_layout(
            value,
            ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        value_list = ttnn.split(value, level_sizes, dim=1)

        sampling_locations = ttnn.to_memory_config(
            sampling_locations,
            ttnn.L1_MEMORY_CONFIG,
        )
        sampling_grids = sampling_locations * 2.0 - 1.0
        sampling_grids = ttnn.permute(sampling_grids, (0, 2, 1, 3, 4, 5))
        sampling_grids = ttnn.reshape(
            sampling_grids,
            (batch_size * num_heads, num_queries, num_levels, num_points, 2),
        )
        sampling_grids = ttnn.to_layout(
            sampling_grids,
            ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        sampling_grid_list = ttnn.split(sampling_grids, 1, dim=2)

        attention_weights = ttnn.to_memory_config(
            attention_weights,
            ttnn.L1_MEMORY_CONFIG,
        )
        attention_weights = ttnn.permute(attention_weights, (0, 2, 1, 3, 4))
        attention_weights = ttnn.reshape(
            attention_weights,
            (batch_size * num_heads, num_queries, num_levels, num_points),
        )
        attention_weights = ttnn.to_layout(
            attention_weights,
            ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        attention_weights_list = ttnn.split(attention_weights, 1, dim=2)

        level_outputs = []
        for level_id, (height, width) in enumerate(value_spatial_shapes_list):
            value_l = ttnn.reshape(
                value_list[level_id],
                (batch_size * num_heads, height, width, hidden_dim),
            )
            sampling_grid_l = sampling_grid_list[level_id]
            sampling_grid_l = ttnn.reshape(
                sampling_grid_l,
                (batch_size * num_heads, num_queries * num_points, 1, 2),
            )

            attention_weights_l = attention_weights_list[level_id]
            attention_weights_l = ttnn.reshape(
                attention_weights_l,
                (batch_size * num_heads, num_queries, num_points, 1),
            )

            sampled_values = ttnn.grid_sample(
                value_l,
                sampling_grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            sampled_values = ttnn.reshape(
                sampled_values,
                (batch_size * num_heads, num_queries, num_points, hidden_dim),
            )

            level_output = sampled_values * attention_weights_l
            level_output = ttnn.sum(level_output, dim=2)
            level_output = ttnn.reshape(level_output, (batch_size * num_heads, num_queries, hidden_dim))
            level_outputs.append(level_output)

        output = level_outputs[0]
        for level_output in level_outputs[1:]:
            output = ttnn.add(
                output,
                level_output,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        output = ttnn.reshape(output, (batch_size, num_heads, num_queries, hidden_dim))
        output = ttnn.permute(output, (0, 2, 1, 3))
        output = ttnn.reshape(output, (batch_size, num_queries, num_heads * hidden_dim))
        output = ttnn.to_layout(
            output,
            ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        return output

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        position_embeddings: ttnn.Tensor,
        reference_points: ttnn.Tensor,
        spatial_shapes: ttnn.Tensor,
        spatial_shapes_list: tuple[tuple[int, int], ...],
        level_start_index: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        hidden_states = ttnn.add(hidden_states, position_embeddings)

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape

        value = ttnn.linear(
            encoder_hidden_states,
            self.value_proj_weight,
            bias=self.value_proj_bias,
        )
        value = ttnn.reshape(
            value,
            (
                batch_size,
                sequence_length,
                self.num_heads,
                self.d_model // self.num_heads,
            ),
        )

        sampling_offsets = ttnn.linear(hidden_states, self.sampling_offsets_weight, bias=self.sampling_offsets_bias)
        sampling_offsets = ttnn.reshape(
            sampling_offsets,
            (batch_size, num_queries, self.num_heads, self.n_levels, self.n_points, 2),
        )

        attention_weights = ttnn.linear(
            hidden_states,
            self.attention_weights_weight,
            bias=self.attention_weights_bias,
        )
        attention_weights = ttnn.reshape(
            attention_weights,
            (batch_size, num_queries, self.num_heads, self.n_levels * self.n_points),
        )
        attention_weights = ttnn.softmax(attention_weights, dim=-1)
        attention_weights = ttnn.reshape(
            attention_weights,
            (batch_size, num_queries, self.num_heads, self.n_levels, self.n_points),
        )

        num_reference_levels = reference_points.shape[-2]
        reference_points_xy = reference_points[..., :2]
        reference_points_wh = reference_points[..., 2:]
        reference_points_xy = ttnn.reshape(
            reference_points_xy,
            (batch_size, num_queries, 1, num_reference_levels, 1, 2),
        )
        reference_points_wh = ttnn.reshape(
            reference_points_wh,
            (batch_size, num_queries, 1, num_reference_levels, 1, 2),
        )

        sampling_locations = ttnn.addcmul(
            reference_points_xy,
            sampling_offsets,
            reference_points_wh,
            value=self.offset_scale / self.n_points,
        )

        output = self._deformable_attention(
            value,
            spatial_shapes,
            spatial_shapes_list,
            level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )

        output = ttnn.linear(
            output,
            self.output_proj_weight,
            bias=self.output_proj_bias,
        )

        return output, attention_weights


class TtRTDetrDecoderLayer:
    def __init__(self, config, parameters, device, dtype):
        self.hidden_size = config.d_model
        self.layer_norm_eps = config.layer_norm_eps

        # Self attention and layer norm
        self.self_attn = TtRTDetrDecoderSelfAttention(
            config, parameters=parameters.self_attn, device=device, dtype=dtype
        )

        self.self_attn_layer_norm_weight = ttnn.from_torch(
            parameters.self_attn_layer_norm.weight,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.self_attn_layer_norm_bias = ttnn.from_torch(
            parameters.self_attn_layer_norm.bias,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # Cross attention and layer norm
        self.encoder_attn = TtRTDetrMultiscaleDeformableAttention(
            config, parameters=parameters.encoder_attn, device=device, dtype=dtype
        )

        self.encoder_attn_layer_norm_weight = ttnn.from_torch(
            parameters.encoder_attn_layer_norm.weight,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.encoder_attn_layer_norm_bias = ttnn.from_torch(
            parameters.encoder_attn_layer_norm.bias,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # MLP and layer norm
        self.mlp = TtRTDetrDecoderMLP(config, parameters=parameters.mlp, device=device, dtype=dtype)

        self.final_layer_norm_weight = ttnn.from_torch(
            parameters.final_layer_norm.weight,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.final_layer_norm_bias = ttnn.from_torch(
            parameters.final_layer_norm.bias,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        object_queries_position_embeddings: ttnn.Tensor,
        reference_points: ttnn.Tensor,
        spatial_shapes: ttnn.Tensor,
        spatial_shapes_list: tuple[tuple[int, int], ...],
        level_start_index: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
    ) -> ttnn.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings=object_queries_position_embeddings,
        )
        hidden_states = ttnn.layer_norm(
            hidden_states,
            epsilon=self.layer_norm_eps,
            weight=self.self_attn_layer_norm_weight,
            bias=self.self_attn_layer_norm_bias,
            residual_input_tensor=residual,
        )

        residual = hidden_states
        hidden_states, _ = self.encoder_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            position_embeddings=object_queries_position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
        )
        hidden_states = ttnn.layer_norm(
            hidden_states,
            epsilon=self.layer_norm_eps,
            weight=self.encoder_attn_layer_norm_weight,
            bias=self.encoder_attn_layer_norm_bias,
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


class TtRTDetrDecoder:
    def __init__(self, config, parameters, device, dtype):
        self.layers = [
            TtRTDetrDecoderLayer(config, parameters=parameters.layers[layer], device=device, dtype=dtype)
            for layer in range(config.decoder_layers)
        ]

        self.query_pos_head = TtRTDetrMLPPredictionHead(
            config,
            parameters=parameters.query_pos_head,
            device=device,
            dtype=dtype,
            input_dim=4,
            hidden_dim=2 * config.d_model,
            output_dim=config.d_model,
            num_layers=2,
        )

        self.bbox_embed = [
            TtRTDetrMLPPredictionHead(
                config,
                parameters=parameters.bbox_embed[index],
                device=device,
                dtype=dtype,
                input_dim=config.d_model,
                hidden_dim=config.d_model,
                output_dim=4,
                num_layers=3,
            )
            for index in range(config.decoder_layers)
        ]

        self.class_embed_weights = [
            ttnn.from_torch(
                parameters.class_embed[index].weight.T,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            for index in range(config.decoder_layers)
        ]

        self.class_embed_biases = [
            ttnn.from_torch(
                parameters.class_embed[index].bias,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            for index in range(config.decoder_layers)
        ]

    def __call__(
        self,
        inputs_embeds: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        reference_points: ttnn.Tensor,
        spatial_shapes: ttnn.Tensor,
        spatial_shapes_list: tuple[tuple[int, int], ...],
        level_start_index: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        hidden_states = inputs_embeds
        intermediate_hidden_states = []
        intermediate_logits = []
        intermediate_reference_points = []

        reference_points_logits = reference_points
        reference_points = ttnn.sigmoid(reference_points_logits)

        for index, decoder_layer in enumerate(self.layers):
            reference_points_input = ttnn.unsqueeze(reference_points, dim=2)
            object_queries_position_embeddings = self.query_pos_head(reference_points)

            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                object_queries_position_embeddings=object_queries_position_embeddings,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                encoder_hidden_states=encoder_hidden_states,
            )

            predicted_corners = self.bbox_embed[index](hidden_states)
            reference_points_logits = ttnn.add(predicted_corners, reference_points_logits)
            reference_points = ttnn.sigmoid(reference_points_logits)

            logits = ttnn.linear(
                hidden_states,
                self.class_embed_weights[index],
                bias=self.class_embed_biases[index],
            )

            intermediate_hidden_states.append(hidden_states)
            intermediate_logits.append(logits)
            intermediate_reference_points.append(reference_points)

        intermediate_hidden_states = ttnn.stack(intermediate_hidden_states, dim=1)
        intermediate_logits = ttnn.stack(intermediate_logits, dim=1)
        intermediate_reference_points = ttnn.stack(intermediate_reference_points, dim=1)

        return (
            hidden_states,
            intermediate_hidden_states,
            intermediate_logits,
            intermediate_reference_points,
        )

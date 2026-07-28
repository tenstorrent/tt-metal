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
        self.num_heads = config.deoder_attention_heads
        self.n_points = config.decoder_n_points

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        reference_points: ttnn.Tensor,
        spatial_shapes: ttnn.Tensor,
        spatial_shapes_list: tuple[tuple[int, int], ...],
        level_start_index: ttnn.Tensor,
    ) -> ttnn.Tensor:
        raise NotImplementedError


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
        self.mlp = TtRTDetrDecoderMLP(config, parameters=parameters.encoder_attn, device=device, dtype=dtype)

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
        raise NotImplementedError


class TtRTDetrDecoder:
    def __init__(self, config, parameters, device, dtype):
        self.layers = [
            TtRTDetrDecoderLayer(config, paraameters=parameters.layers[layer], device=device, dtype=dtype)
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
            num_layers=2.0,
        )

    def __call__(
        self,
        inputs_embeds: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        reference_points: ttnn.Tensor,
        spatial_shapes: ttnn.Tensor,
        spatial_shapes_list: tuple[tuple[int, int], ...],
        level_start_index: ttnn.Tensor,
    ):
        raise NotImplementedError

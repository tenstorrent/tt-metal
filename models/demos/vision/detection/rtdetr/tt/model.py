import torch

import ttnn
from models.demos.vision.detection.rtdetr.tt.backbone import TtRTDetrConvEncoder, TtRTDetrResNetConvLayer
from models.demos.vision.detection.rtdetr.tt.decoder import TtRTDetrDecoder
from models.demos.vision.detection.rtdetr.tt.encoder import TtRTDetrHybridEncoder


class TtRTDetrModel:
    def __init__(self, config, parameters, device, dtype, input_height=640, input_width=640):
        self.device = device
        self.dtype = dtype

        self.backbone = TtRTDetrConvEncoder(config, parameters=parameters.backbone, device=device, dtype=dtype)

        self.encoder_in_channels = config.encoder_in_channels
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.decoder_in_channels = config.decoder_in_channels
        self.decoder_hidden_dim = config.d_model
        self.num_queries = config.num_queries
        self.layer_norm_eps = config.layer_norm_eps

        self.encoder_input_proj = [
            TtRTDetrResNetConvLayer(
                config=config,
                parameters=parameters.encoder_input_proj[level],
                device=device,
                dtype=dtype,
                in_channels=in_channels,
                out_channels=self.encoder_hidden_dim,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                activation="identity",
            )
            for level, in_channels in enumerate(self.encoder_in_channels)
        ]

        self.encoder = TtRTDetrHybridEncoder(config=config, parameters=parameters.encoder, device=device, dtype=dtype)

        self.decoder_input_proj = [
            TtRTDetrResNetConvLayer(
                config=config,
                parameters=parameters.decoder_input_proj[level],
                device=device,
                dtype=dtype,
                in_channels=in_channels,
                out_channels=config.d_model,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                activation="identity",
            )
            for level, in_channels in enumerate(self.decoder_in_channels)
        ]

        self.decoder = TtRTDetrDecoder(config=config, parameters=parameters.decoder, device=device, dtype=dtype)

        self.spatial_shapes_list = tuple(
            (input_height // stride, input_width // stride) for stride in config.feat_strides
        )

        self.spatial_shapes_tensor = ttnn.from_torch(
            torch.tensor(self.spatial_shapes_list, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        level_sizes = [height * width for height, width in self.spatial_shapes_list]

        level_start_index = [0]
        for level_size in level_sizes[:-1]:
            level_start_index.append(level_start_index[-1] + level_size)

        self.level_start_index = ttnn.from_torch(
            torch.tensor(level_start_index, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Since fixed input size, can generate anchors at init with cpu
        self.anchors, self.valid_mask = self.generate_anchors(self.spatial_shapes_list)

        self.enc_output_weight = ttnn.from_torch(
            parameters.enc_output[0].weight.T,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.enc_output_bias = ttnn.from_torch(
            parameters.enc_output[0].bias,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.enc_output_norm_weight = ttnn.from_torch(
            parameters.enc_output[1].weight,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.enc_output_norm_bias = ttnn.from_torch(
            parameters.enc_output[1].bias,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.enc_score_head_weight = ttnn.from_torch(
            parameters.enc_score_head.weight.T,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.enc_score_head_bias = ttnn.from_torch(
            parameters.enc_score_head.bias,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.enc_bbox_head_0_weight = ttnn.from_torch(
            parameters.enc_bbox_head.layers[0].weight.T,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.enc_bbox_head_0_bias = ttnn.from_torch(
            parameters.enc_bbox_head.layers[0].bias,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.enc_bbox_head_1_weight = ttnn.from_torch(
            parameters.enc_bbox_head.layers[1].weight.T,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.enc_bbox_head_1_bias = ttnn.from_torch(
            parameters.enc_bbox_head.layers[1].bias,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.enc_bbox_head_2_weight = ttnn.from_torch(
            parameters.enc_bbox_head.layers[2].weight.T,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.enc_bbox_head_2_bias = ttnn.from_torch(
            parameters.enc_bbox_head.layers[2].bias,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def generate_anchors(self, spatial_shapes, grid_size=0.05):
        if self.dtype == ttnn.bfloat16:
            torch_dtype = torch.bfloat16
        elif self.dtype == ttnn.float32:
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported anchor dtype: {self.dtype}")

        anchors = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, dtype=torch_dtype),
                torch.arange(width, dtype=torch_dtype),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0) + 0.5
            grid_xy[..., 0] /= width
            grid_xy[..., 1] /= height

            anchor_wh = torch.ones_like(grid_xy) * grid_size * (2.0**level)
            anchors.append(torch.concat([grid_xy, anchor_wh], dim=-1).reshape(1, height * width, 4))

        anchors = torch.concat(anchors, dim=1)
        valid_mask = ((anchors > 0.01) & (anchors < 0.99)).all(dim=-1, keepdim=True)
        anchors = torch.log(anchors / (1.0 - anchors))
        anchors = torch.where(
            valid_mask,
            anchors,
            torch.full_like(anchors, torch.finfo(torch_dtype).max),
        )

        anchors = ttnn.from_torch(
            anchors,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        valid_mask = ttnn.from_torch(
            valid_mask.to(torch_dtype),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return anchors, valid_mask

    def __call__(self, pixel_values: ttnn.Tensor):
        batch_size = pixel_values.shape[0]

        # Get features from backbone
        features = self.backbone(pixel_values)

        # Project features for the encoder
        projected_features = []
        for (feature, height, width), projection in zip(features, self.encoder_input_proj):
            feature, height, width = projection(
                x=feature,
                batch_size=batch_size,
                input_height=height,
                input_width=width,
            )
            projected_features.append((feature, height, width))

        # Encoder
        encoder_outputs = self.encoder(inputs_embeds=projected_features, batch_size=batch_size)

        # Project encoder outputs for decoder
        sources = []
        for (source, height, width), projection in zip(encoder_outputs, self.decoder_input_proj):
            source, height, width = projection(
                x=source,
                batch_size=batch_size,
                input_height=height,
                input_width=width,
            )
            sources.append((source, height, width))

        source_flatten = []
        for source, height, width in sources:
            source = ttnn.to_memory_config(source, ttnn.DRAM_MEMORY_CONFIG)
            source = ttnn.reshape(
                source,
                (batch_size, height * width, self.decoder_hidden_dim),
            )
            source_flatten.append(source)

        source_flatten = ttnn.concat(source_flatten, dim=1)
        source_flatten = ttnn.to_layout(source_flatten, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # use the valid_mask to selectively retain values in the feature map where the mask is `True`
        memory = ttnn.multiply(self.valid_mask, source_flatten)

        # enc_output
        output_memory = ttnn.layer_norm(
            input_tensor=ttnn.linear(memory, self.enc_output_weight, self.enc_output_bias),
            epsilon=self.layer_norm_eps,
            weight=self.enc_output_norm_weight,
            bias=self.enc_output_norm_bias,
        )

        # enc_score_head
        enc_outputs_class = ttnn.linear(output_memory, self.enc_score_head_weight, self.enc_score_head_bias)

        # enc_bbox_head
        enc_outputs_coord_logits = ttnn.linear(
            output_memory, self.enc_bbox_head_0_weight, bias=self.enc_bbox_head_0_bias, activation="relu"
        )

        enc_outputs_coord_logits = ttnn.linear(
            enc_outputs_coord_logits, self.enc_bbox_head_1_weight, bias=self.enc_bbox_head_1_bias, activation="relu"
        )

        enc_outputs_coord_logits = ttnn.linear(
            enc_outputs_coord_logits, self.enc_bbox_head_2_weight, bias=self.enc_bbox_head_2_bias
        )

        enc_outputs_coord_logits = ttnn.add(enc_outputs_coord_logits, self.anchors)

        enc_outputs_class_max = ttnn.max(enc_outputs_class, dim=-1, keepdim=False)
        _, topk_ind = ttnn.topk(enc_outputs_class_max, self.num_queries, dim=1)

        reference_points_unact = ttnn.gather(
            enc_outputs_coord_logits,
            dim=1,
            index=ttnn.repeat(ttnn.unsqueeze(topk_ind, dim=-1), (1, 1, enc_outputs_coord_logits.shape[-1])),
        )

        enc_topk_bboxes = ttnn.sigmoid(reference_points_unact)

        enc_topk_logits = ttnn.gather(
            enc_outputs_class,
            dim=1,
            index=ttnn.repeat(ttnn.unsqueeze(topk_ind, dim=-1), (1, 1, enc_outputs_class.shape[-1])),
        )

        target = ttnn.gather(
            output_memory, dim=1, index=ttnn.repeat(ttnn.unsqueeze(topk_ind, dim=-1), (1, 1, output_memory.shape[-1]))
        )

        decoder_outputs = self.decoder(
            inputs_embeds=target,
            encoder_hidden_states=source_flatten,
            reference_points=reference_points_unact,
            spatial_shapes=self.spatial_shapes_tensor,
            spatial_shapes_list=self.spatial_shapes_list,
            level_start_index=self.level_start_index,
        )

        return decoder_outputs

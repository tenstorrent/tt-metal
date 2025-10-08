# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.transfuser.tt.utils import TTConv2D
from typing import List
from loguru import logger
from models.experimental.transfuser.tt.bottleneck import TTRegNetBottleneck


class TtTransfuserBackbone:
    def __init__(
        self,
        device,
        parameters,
        stride,
        model_config,
        config,
    ) -> None:
        self.device = device
        self.config = config
        self.inplanes = 32
        self.conv1 = TTConv2D(
            kernel_size=3,
            stride=2,
            padding=1,
            parameters=parameters.image_encoder.features.conv1,
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            dtype=ttnn.bfloat16,
        )
        self.lidar_conv1 = TTConv2D(
            kernel_size=3,
            stride=2,
            padding=1,
            parameters=parameters.lidar_encoder._model.conv1,
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            dtype=ttnn.bfloat16,
        )
        # Layer1 for both encoders
        self.image_layer1 = self._make_layer(
            parameters=parameters.image_encoder.features.layer1,
            planes=72,
            blocks=2,  # b1 and b2
            stride=2,
            groups=3,  # conv2
            model_config=model_config,
            stage_name="layer1",
        )

        self.lidar_layer1 = self._make_layer(
            parameters=parameters.lidar_encoder._model.layer1,
            planes=72,
            blocks=2,
            stride=2,
            groups=3,
            model_config=model_config,
            stage_name="layer1",
        )

        # self.transformer1 = TTGpt(
        #     device=self.device,
        #     parameters=parameters["transformer1"],
        #     n_head=config.n_head,
        #     n_layer=config.n_layer,
        #     use_velocity=config.use_velocity,
        #     img_vert_anchors=config.img_vert_anchors,
        #     img_horz_anchors=config.img_horz_anchors,
        #     lidar_vert_anchors=config.lidar_vert_anchors,
        #     lidar_horz_anchors=config.lidar_horz_anchors,
        #     seq_len=config.seq_len,
        #     n_embd=72,  # layer1 output channels
        #     dtype=ttnn.bfloat16,
        #     memory_config=ttnn.L1_MEMORY_CONFIG,
        # )

    def _make_layer(
        self,
        parameters,
        planes: int,
        blocks: int,
        stride: int,
        groups: int = 1,
        model_config=None,
        stage_name=None,
    ) -> List[TTRegNetBottleneck]:
        layers = []

        # Determine shard layout based on stage name
        if stage_name == "layer1":
            shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        elif stage_name == "layer2":
            shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        elif stage_name == "layer3":
            shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        elif stage_name == "layer4":
            shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        else:
            # Default to HEIGHT_SHARDED for backward compatibility
            shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

        # First block (may have downsample)
        downsample = stride != 1 or self.inplanes != planes
        layers.append(
            TTRegNetBottleneck(
                parameters=parameters["b1"],
                model_config=model_config,
                stride=stride,
                downsample=downsample,
                groups=groups,
                shard_layout=shard_layout,
            )
        )
        self.inplanes = planes

        # Remaining blocks
        for block_num in range(1, blocks):
            block_name = f"b{block_num + 1}"
            layers.append(
                TTRegNetBottleneck(
                    parameters=parameters[block_name],
                    model_config=model_config,
                    stride=1,
                    downsample=False,
                    groups=groups,
                    shard_layout=shard_layout,
                )
            )

        return layers

    def normalize_imagenet_ttnn(self, x):
        """Normalize input images according to ImageNet standards using TTNN operations.
        Expects input in NHWC format
        """
        # First divide by 255.0 to convert from [0,255] to [0,1]
        x = ttnn.multiply(x, 1.0 / 255.0)

        # For NHWC format: [batch, height, width, channels]
        # Slice along the channel dimension (dim=3)
        x_r = ttnn.slice(x, [0, 0, 0, 0], [x.shape[0], x.shape[1], x.shape[2], 1])  # Red channel
        x_g = ttnn.slice(x, [0, 0, 0, 1], [x.shape[0], x.shape[1], x.shape[2], 2])  # Green channel
        x_b = ttnn.slice(x, [0, 0, 0, 2], [x.shape[0], x.shape[1], x.shape[2], 3])  # Blue channel

        # Normalize each channel: (x - mean) / std
        x_r = ttnn.subtract(x_r, 0.485)
        x_r = ttnn.multiply(x_r, 1.0 / 0.229)

        x_g = ttnn.subtract(x_g, 0.456)
        x_g = ttnn.multiply(x_g, 1.0 / 0.224)

        x_b = ttnn.subtract(x_b, 0.406)
        x_b = ttnn.multiply(x_b, 1.0 / 0.225)

        # Concatenate along channel dimension (dim=3 for NHWC)
        x = ttnn.concat([x_r, x_g, x_b], dim=3)

        return x

    def __call__(self, image_x, lidar_x, device):
        # Process image input
        logger.info(f"image_encoder_conv1")
        image_x = self.normalize_imagenet_ttnn(image_x)
        image_out, image_shape = self.conv1(device, image_x, image_x.shape)
        # Reshape to spatial dimensions: 80 * 352 = 28160
        # out = ttnn.reshape(out, (1, 80, 352, 32))
        # out = ttnn.permute(out, (0, 3, 1, 2))
        logger.info(f"lidar_encoder_conv1")
        # Process lidar input
        lidar_out, lidar_shape = self.lidar_conv1(device, lidar_x, lidar_x.shape)
        print("..........................................")
        print(lidar_shape)
        print(image_shape)

        logger.info(f"image_encoder_layer1")
        # image_out = ttnn.reshape(image_out, (1, 80, 352, 32))
        image_out = ttnn.reshape(image_out, image_shape)
        # Process layer1 blocks
        for block in self.image_layer1:
            image_out = block(image_out, device)

        logger.info(f"lidar_encoder_layer1")
        lidar_out = ttnn.reshape(lidar_out, lidar_shape)
        # lidar_out = ttnn.reshape(lidar_out, (1, 128, 128, 32))
        for block in self.lidar_layer1:
            lidar_out = block(lidar_out, device)

        logger.info(f"img_avgpool")

        image_h = image_out.shape[1]
        image_w = image_out.shape[2]
        image_c = image_out.shape[3]

        image_features_flat = ttnn.reshape(image_out, (1, 1, image_out.shape[0] * image_h * image_w, image_c))
        image_embd_layer1 = ttnn.adaptive_avg_pool2d(
            input_tensor=image_features_flat,
            batch_size=image_out.shape[0],
            input_h=image_h,
            input_w=image_w,
            channels=image_c,
            output_size=[self.config.img_vert_anchors, self.config.img_horz_anchors],
        )
        logger.info(f"lidar_avgpool")
        lidar_h = lidar_out.shape[1]
        lidar_w = lidar_out.shape[2]
        lidar_c = lidar_out.shape[3]

        lidar_features_flat = ttnn.reshape(lidar_out, (1, 1, lidar_out.shape[0] * lidar_h * lidar_w, lidar_c))

        lidar_embd_layer1 = ttnn.adaptive_avg_pool2d(
            input_tensor=lidar_features_flat,
            batch_size=lidar_out.shape[0],
            input_h=lidar_h,
            input_w=lidar_w,
            channels=lidar_c,
            output_size=[self.config.lidar_vert_anchors, self.config.lidar_horz_anchors],
        )

        return image_embd_layer1, lidar_embd_layer1

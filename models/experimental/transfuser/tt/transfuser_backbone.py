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
        parameters,
        stride,
        model_config,
        # layer_optimisations=neck_optimisations,
    ) -> None:
        self.inplanes = 32
        # print(f"{parameters=}")
        self.conv1 = TTConv2D(
            kernel_size=3,
            stride=2,
            padding=1,
            parameters=parameters.image_encoder.features.conv1,
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceHeight, num_slices=2),
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
        )

        self.lidar_layer1 = self._make_layer(
            parameters=parameters.lidar_encoder._model.layer1,
            planes=72,
            blocks=2,
            stride=2,
            groups=3,
            model_config=model_config,
        )

    def _make_layer(
        self,
        parameters,
        planes: int,
        blocks: int,
        stride: int,
        groups: int = 1,
        model_config=None,
    ) -> List[TTRegNetBottleneck]:
        layers = []

        # First block (may have downsample)
        downsample = stride != 1 or self.inplanes != planes
        layers.append(
            TTRegNetBottleneck(
                parameters=parameters["b1"],
                model_config=model_config,
                stride=stride,
                downsample=downsample,
                groups=groups,
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
                )
            )

        return layers

    def normalize_imagenet_ttnn(self, x):
        """Normalize input images according to ImageNet standards using TTNN operations."""
        # ImageNet normalization constants
        # Channel 0 (R): (x/255.0 - 0.485) / 0.229
        # Channel 1 (G): (x/255.0 - 0.456) / 0.224
        # Channel 2 (B): (x/255.0 - 0.406) / 0.225

        # First divide by 255.0 to convert from [0,255] to [0,1]
        x = ttnn.multiply(x, 1.0 / 255.0)

        # Split channels for per-channel normalization
        # Note: x is in NCHW format at this point
        x_r = ttnn.slice(x, [0, 0, 0, 0], [x.shape[0], 1, x.shape[2], x.shape[3]])  # Red channel
        x_g = ttnn.slice(x, [0, 1, 0, 0], [x.shape[0], 2, x.shape[2], x.shape[3]])  # Green channel
        x_b = ttnn.slice(x, [0, 2, 0, 0], [x.shape[0], 3, x.shape[2], x.shape[3]])  # Blue channel

        # Normalize each channel: (x - mean) / std
        x_r = ttnn.subtract(x_r, 0.485)
        x_r = ttnn.multiply(x_r, 1.0 / 0.229)

        x_g = ttnn.subtract(x_g, 0.456)
        x_g = ttnn.multiply(x_g, 1.0 / 0.224)

        x_b = ttnn.subtract(x_b, 0.406)
        x_b = ttnn.multiply(x_b, 1.0 / 0.225)

        # Concatenate channels back together
        x = ttnn.concat([x_r, x_g, x_b], dim=1)

        return x

    def __call__(self, image_x, lidar_x, device):
        # Process image input
        logger.info(f"image_encoder_conv1")
        image_x = self.normalize_imagenet_ttnn(image_x)
        print("///////////////////")
        print(image_x.shape)
        print(lidar_x.shape)
        # image_x = ttnn.permute(image_x, (0, 2, 3, 1))
        image_out, image_shape = self.conv1(device, image_x, image_x.shape)
        # Reshape to spatial dimensions: 80 * 352 = 28160
        # out = ttnn.reshape(out, (1, 80, 352, 32))
        # out = ttnn.permute(out, (0, 3, 1, 2))
        logger.info(f"lidar_encoder_conv1")
        # Process lidar input
        # lidar_x = ttnn.permute(lidar_x, (0, 2, 3, 1))
        lidar_out, lidar_shape = self.lidar_conv1(device, lidar_x, lidar_x.shape)
        print("..........................................")
        print(lidar_shape)
        print(image_shape)

        logger.info(f"image_encoder_layer1")
        image_out = ttnn.reshape(image_out, (1, 80, 352, 32))
        # Process layer1 blocks
        for block in self.image_layer1:
            image_out = block(image_out, device)

        logger.info(f"lidar_encoder_layer1")
        lidar_out = ttnn.reshape(lidar_out, (1, 128, 128, 32))
        for block in self.lidar_layer1:
            lidar_out = block(lidar_out, device)

        return image_out, lidar_out

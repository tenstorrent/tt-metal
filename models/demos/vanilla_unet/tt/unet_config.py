# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import ttnn
from models.tt_cnn.tt.builder import (
    AutoShardedStrategyConfiguration,
    Conv2dConfiguration,
    HeightShardedStrategyConfiguration,
    MaxPool2dConfiguration,
)


@dataclass
class UpconvConfiguration:
    """Configuration for transpose convolution (upsampling) operations"""

    input_height: int
    input_width: int
    in_channels: int
    out_channels: int
    batch_size: int
    kernel_size: Tuple[int, int] = (2, 2)
    stride: Tuple[int, int] = (2, 2)
    padding: Tuple[int, int] = (0, 0)
    weight: ttnn.Tensor = None
    bias: ttnn.Tensor = None


@dataclass
class TtUNetLayerConfigs:
    encoder1_conv1: Conv2dConfiguration
    encoder1_conv2: Conv2dConfiguration
    encoder1_pool: MaxPool2dConfiguration

    encoder2_conv1: Conv2dConfiguration
    encoder2_conv2: Conv2dConfiguration
    encoder2_pool: MaxPool2dConfiguration

    encoder3_conv1: Conv2dConfiguration
    encoder3_conv2: Conv2dConfiguration
    encoder3_pool: MaxPool2dConfiguration

    encoder4_conv1: Conv2dConfiguration
    encoder4_conv2: Conv2dConfiguration
    encoder4_pool: MaxPool2dConfiguration

    bottleneck_conv1: Conv2dConfiguration
    bottleneck_conv2: Conv2dConfiguration

    decoder4_conv1: Conv2dConfiguration
    decoder4_conv2: Conv2dConfiguration
    upconv4: UpconvConfiguration

    decoder3_conv1: Conv2dConfiguration
    decoder3_conv2: Conv2dConfiguration
    upconv3: UpconvConfiguration

    decoder2_conv1: Conv2dConfiguration
    decoder2_conv2: Conv2dConfiguration
    upconv2: UpconvConfiguration

    decoder1_conv1: Conv2dConfiguration
    decoder1_conv2: Conv2dConfiguration
    upconv1: UpconvConfiguration

    final_conv: Conv2dConfiguration


class TtUNetConfigBuilder:
    def __init__(
        self,
        parameters: Dict,
        in_channels: int = 3,
        out_channels: int = 1,
        init_features: int = 32,
        input_height: int = 480,
        input_width: int = 640,
        batch_size: int = 1,
        sharding_strategy: Optional[HeightShardedStrategyConfiguration] = None,
    ):
        self.parameters = parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features
        self.input_height = input_height
        self.input_width = input_width
        self.batch_size = batch_size

        self.default_sharding_strategy = HeightShardedStrategyConfiguration(act_block_h_override=32)

        self.features = init_features

    def build_configs(self) -> TtUNetLayerConfigs:
        """Build all layer configurations from the preprocessed parameters"""
        return TtUNetLayerConfigs(
            # Encoder 1
            encoder1_conv1=self._create_conv_config_from_params(
                self.input_height,
                self.input_width,
                self.in_channels,
                self.features,
                self.parameters["encoder1"][0],
            ),
            encoder1_conv2=self._create_conv_config_from_params(
                self.input_height,
                self.input_width,
                self.features,
                self.features,
                self.parameters["encoder1"][1],
                activation_dtype=ttnn.bfloat8_b,
                output_dtype=ttnn.bfloat8_b,
            ),
            encoder1_pool=self._create_pool_config(self.input_height, self.input_width, self.features),
            # Encoder 2
            encoder2_conv1=self._create_conv_config_from_params(
                self.input_height // 2,
                self.input_width // 2,
                self.features,
                self.features * 2,
                self.parameters["encoder2"][0],
            ),
            encoder2_conv2=self._create_conv_config_from_params(
                self.input_height // 2,
                self.input_width // 2,
                self.features * 2,
                self.features * 2,
                self.parameters["encoder2"][1],
                deallocate_activation=False,
            ),
            encoder2_pool=self._create_pool_config(self.input_height // 2, self.input_width // 2, self.features * 2),
            # Encoder 3
            encoder3_conv1=self._create_conv_config_from_params(
                self.input_height // 4,
                self.input_width // 4,
                self.features * 2,
                self.features * 4,
                self.parameters["encoder3"][0],
            ),
            encoder3_conv2=self._create_conv_config_from_params(
                self.input_height // 4,
                self.input_width // 4,
                self.features * 4,
                self.features * 4,
                self.parameters["encoder3"][1],
            ),
            encoder3_pool=self._create_pool_config(self.input_height // 4, self.input_width // 4, self.features * 4),
            # Encoder 4
            encoder4_conv1=self._create_conv_config_from_params(
                self.input_height // 8,
                self.input_width // 8,
                self.features * 4,
                self.features * 8,
                self.parameters["encoder4"][0],
            ),
            encoder4_conv2=self._create_conv_config_from_params(
                self.input_height // 8,
                self.input_width // 8,
                self.features * 8,
                self.features * 8,
                self.parameters["encoder4"][1],
            ),
            encoder4_pool=self._create_pool_config(self.input_height // 8, self.input_width // 8, self.features * 8),
            # Bottleneck
            bottleneck_conv1=self._create_conv_config_from_params(
                self.input_height // 16,
                self.input_width // 16,
                self.features * 8,
                self.features * 16,
                self.parameters["bottleneck"][0],
            ),
            bottleneck_conv2=self._create_conv_config_from_params(
                self.input_height // 16,
                self.input_width // 16,
                self.features * 16,
                self.features * 16,
                self.parameters["bottleneck"][1],
            ),
            # Decoder 4 (after upconv: features*16 -> features*8, concat with skip: features*8+features*8 -> features*8)
            decoder4_conv1=self._create_conv_config_from_params(
                self.input_height // 8,
                self.input_width // 8,
                self.features * 16,
                self.features * 8,  # After concat: features*8 + features*8
                self.parameters["decoder4"][0],
                sharding_strategy=AutoShardedStrategyConfiguration(),  # WidthShardedStrategyConfiguration(reshard_if_not_optimal=True, act_block_w_div=3),
                activation_dtype=ttnn.bfloat8_b,
                output_dtype=ttnn.bfloat8_b,
            ),
            decoder4_conv2=self._create_conv_config_from_params(
                self.input_height // 8,
                self.input_width // 8,
                self.features * 8,
                self.features * 8,
                self.parameters["decoder4"][1],
            ),
            # Decoder 3
            decoder3_conv1=self._create_conv_config_from_params(
                self.input_height // 4,
                self.input_width // 4,
                self.features * 8,
                self.features * 4,  # After concat: features*4 + features*4
                self.parameters["decoder3"][0],
            ),
            decoder3_conv2=self._create_conv_config_from_params(
                self.input_height // 4,
                self.input_width // 4,
                self.features * 4,
                self.features * 4,
                self.parameters["decoder3"][1],
            ),
            # Decoder 2
            decoder2_conv1=self._create_conv_config_from_params(
                self.input_height // 2,
                self.input_width // 2,
                self.features * 4,
                self.features * 2,  # After concat: features*2 + features*2
                self.parameters["decoder2"][0],
            ),
            decoder2_conv2=self._create_conv_config_from_params(
                self.input_height // 2,
                self.input_width // 2,
                self.features * 2,
                self.features * 2,
                self.parameters["decoder2"][1],
                activation_dtype=ttnn.bfloat8_b,
                output_dtype=ttnn.bfloat8_b,
            ),
            # Decoder 1
            decoder1_conv1=self._create_conv_config_from_params(
                self.input_height,
                self.input_width,
                self.features * 2,
                self.features,  # After concat: features + features
                self.parameters["decoder1"][0],
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            decoder1_conv2=self._create_conv_config_from_params(
                self.input_height,
                self.input_width,
                self.features,
                self.features,
                self.parameters["decoder1"][1],
                activation_dtype=ttnn.bfloat8_b,
                output_dtype=ttnn.bfloat8_b,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            # Final convolution (1x1 conv)
            final_conv=self._create_conv_config_from_params(
                self.input_height,
                self.input_width,
                self.features,
                self.out_channels,
                self.parameters["conv"],
                kernel_size=(1, 1),
                padding=(0, 0),
                weights_dtype=ttnn.bfloat16,
                math_fidelity=ttnn.MathFidelity.HiFi2,
                activation=None,  # No activation for final layer
            ),
            # Transpose convolution configurations from preprocessed parameters
            upconv4=self._create_upconv_config(
                self.input_height // 16,
                self.input_width // 16,
                self.features * 16,
                self.features * 8,
                self.parameters["upconv4"]["weight"],
                self.parameters["upconv4"]["bias"],
            ),
            upconv3=self._create_upconv_config(
                self.input_height // 8,
                self.input_width // 8,
                self.features * 8,
                self.features * 4,
                self.parameters["upconv3"]["weight"],
                self.parameters["upconv3"]["bias"],
            ),
            upconv2=self._create_upconv_config(
                self.input_height // 4,
                self.input_width // 4,
                self.features * 4,
                self.features * 2,
                self.parameters["upconv2"]["weight"],
                self.parameters["upconv2"]["bias"],
            ),
            upconv1=self._create_upconv_config(
                self.input_height // 2,
                self.input_width // 2,
                self.features * 2,
                self.features,
                self.parameters["upconv1"]["weight"],
                self.parameters["upconv1"]["bias"],
            ),
        )

    def _create_conv_config_from_params(
        self,
        input_height: int,
        input_width: int,
        in_channels: int,
        out_channels: int,
        parameters: Dict,
        kernel_size: Tuple[int, int] = (3, 3),
        padding: Tuple[int, int] = (1, 1),
        activation: Optional[ttnn.UnaryWithParam] = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        deallocate_activation: bool = True,
        sharding_strategy=None,
        activation_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        output_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    ) -> Conv2dConfiguration:
        """Create a Conv2dConfiguration from preprocessed parameters dict containing weight and bias"""
        return Conv2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=self.batch_size,
            kernel_size=kernel_size,
            padding=padding,
            weight=parameters["weight"],
            bias=parameters["bias"],
            activation=activation,
            deallocate_activation=deallocate_activation,
            sharding_strategy=sharding_strategy if sharding_strategy is not None else self.default_sharding_strategy,
            activation_dtype=activation_dtype,
            output_dtype=output_dtype,
            weights_dtype=weights_dtype,
            output_layout=ttnn.TILE_LAYOUT,
            enable_weights_double_buffer=False,
            enable_act_double_buffer=False,
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=packer_l1_acc,
        )

    def _create_pool_config(self, input_height: int, input_width: int, channels: int) -> MaxPool2dConfiguration:
        return MaxPool2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            channels=channels,
            batch_size=self.batch_size,
            kernel_size=(2, 2),
            stride=(2, 2),
        )

    def _create_upconv_config(
        self,
        input_height: int,
        input_width: int,
        in_channels: int,
        out_channels: int,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
    ) -> UpconvConfiguration:
        """Create an UpconvConfiguration from parameters"""
        return UpconvConfiguration(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=self.batch_size,
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
            weight=weight,
            bias=bias,
        )


def create_unet_configs_from_parameters(
    parameters: Dict,
    in_channels: int = 3,
    out_channels: int = 1,
    init_features: int = 32,
    input_height: int = 480,
    input_width: int = 640,
    batch_size: int = 1,
) -> TtUNetLayerConfigs:
    builder = TtUNetConfigBuilder(
        parameters=parameters,
        in_channels=in_channels,
        out_channels=out_channels,
        init_features=init_features,
        input_height=input_height,
        input_width=input_width,
        batch_size=batch_size,
    )

    return builder.build_configs()

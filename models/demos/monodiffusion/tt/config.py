# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import ttnn
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    HeightShardedStrategyConfiguration,
    MaxPool2dConfiguration,
    ShardingStrategy,
)


@dataclass
class TtMonoDiffusionLayerConfigs:
    """Complete layer configurations for MonoDiffusion model"""
    # Encoder configurations
    encoder_conv1: Conv2dConfiguration
    encoder_conv2: Conv2dConfiguration
    encoder_conv3: Conv2dConfiguration
    encoder_conv4: Conv2dConfiguration
    encoder_pool1: MaxPool2dConfiguration
    encoder_pool2: MaxPool2dConfiguration

    # Diffusion U-Net configurations
    unet_down1_conv1: Conv2dConfiguration
    unet_down1_conv2: Conv2dConfiguration
    unet_mid_conv1: Conv2dConfiguration
    unet_mid_conv2: Conv2dConfiguration
    unet_up1_conv1: Conv2dConfiguration
    unet_up1_conv2: Conv2dConfiguration

    # Decoder configurations
    decoder_conv1: Conv2dConfiguration
    decoder_conv2: Conv2dConfiguration
    decoder_conv3: Conv2dConfiguration
    decoder_conv4: Conv2dConfiguration
    final_depth_conv: Conv2dConfiguration

    # Uncertainty head configurations
    uncertainty_conv1: Conv2dConfiguration
    uncertainty_conv2: Conv2dConfiguration

    # Timestep embedding dimension
    timestep_embed_dim: int = 256
    num_inference_steps: int = 20


class TtMonoDiffusionConfigBuilder:
    """Builder for MonoDiffusion configurations following vanilla_unet pattern"""

    def __init__(
        self,
        parameters: Dict,
        batch_size: int = 1,
        input_height: int = 192,
        input_width: int = 640,
        in_channels: int = 3,
    ):
        self.parameters = parameters
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.in_channels = in_channels

    def _create_conv_config(
        self,
        input_height: int,
        input_width: int,
        in_channels: int,
        out_channels: int,
        parameters: Dict,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (1, 1),
        activation: Optional[ttnn.UnaryWithParam] = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        sharding_strategy: ShardingStrategy = HeightShardedStrategyConfiguration(act_block_h_override=32),
        **kwargs
    ) -> Conv2dConfiguration:
        """Create Conv2dConfiguration following vanilla_unet pattern"""
        return Conv2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=self.batch_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            weight=parameters["weight"],
            bias=parameters["bias"],
            activation=activation,
            sharding_strategy=sharding_strategy,
            activation_dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat8_b,
            output_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.LoFi,
            deallocate_activation=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            output_layout=ttnn.TILE_LAYOUT,
            **kwargs
        )

    def _create_pool_config(
        self,
        input_height: int,
        input_width: int,
        channels: int
    ) -> MaxPool2dConfiguration:
        """Create MaxPool2dConfiguration"""
        return MaxPool2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            channels=channels,
            batch_size=self.batch_size,
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
        )

    def build_configs(self) -> TtMonoDiffusionLayerConfigs:
        """Build complete layer configurations"""
        return TtMonoDiffusionLayerConfigs(
            # Encoder
            encoder_conv1=self._create_conv_config(
                self.input_height, self.input_width, self.in_channels, 64,
                self.parameters["encoder"]["conv1"],
                kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=6 * 32),
            ),
            encoder_conv2=self._create_conv_config(
                self.input_height // 2, self.input_width // 2, 64, 128,
                self.parameters["encoder"]["conv2"],
                sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=3 * 32),
            ),
            encoder_conv3=self._create_conv_config(
                self.input_height // 4, self.input_width // 4, 128, 256,
                self.parameters["encoder"]["conv3"],
                sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=2 * 32),
            ),
            encoder_conv4=self._create_conv_config(
                self.input_height // 8, self.input_width // 8, 256, 512,
                self.parameters["encoder"]["conv4"],
                sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=32),
            ),
            encoder_pool1=self._create_pool_config(self.input_height // 2, self.input_width // 2, 64),
            encoder_pool2=self._create_pool_config(self.input_height // 4, self.input_width // 4, 128),

            # Diffusion U-Net
            unet_down1_conv1=self._create_conv_config(
                self.input_height // 16, self.input_width // 16, 512, 512,
                self.parameters["unet"]["down1_conv1"],
                sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=32),
            ),
            unet_down1_conv2=self._create_conv_config(
                self.input_height // 16, self.input_width // 16, 512, 512,
                self.parameters["unet"]["down1_conv2"],
                sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=32),
            ),
            unet_mid_conv1=self._create_conv_config(
                self.input_height // 32, self.input_width // 32, 512, 512,
                self.parameters["unet"]["mid_conv1"],
                sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=32),
            ),
            unet_mid_conv2=self._create_conv_config(
                self.input_height // 32, self.input_width // 32, 512, 512,
                self.parameters["unet"]["mid_conv2"],
                sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=32),
            ),
            unet_up1_conv1=self._create_conv_config(
                self.input_height // 16, self.input_width // 16, 1024, 512,
                self.parameters["unet"]["up1_conv1"],
                sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=32),
            ),
            unet_up1_conv2=self._create_conv_config(
                self.input_height // 16, self.input_width // 16, 512, 512,
                self.parameters["unet"]["up1_conv2"],
                sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=32),
            ),

            # Decoder
            decoder_conv1=self._create_conv_config(
                self.input_height // 8, self.input_width // 8, 512, 256,
                self.parameters["decoder"]["conv1"],
                sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=2 * 32),
            ),
            decoder_conv2=self._create_conv_config(
                self.input_height // 4, self.input_width // 4, 256, 128,
                self.parameters["decoder"]["conv2"],
                sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=3 * 32),
            ),
            decoder_conv3=self._create_conv_config(
                self.input_height // 2, self.input_width // 2, 128, 64,
                self.parameters["decoder"]["conv3"],
                sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=5 * 32),
            ),
            decoder_conv4=self._create_conv_config(
                self.input_height, self.input_width, 64, 32,
                self.parameters["decoder"]["conv4"],
                sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=10 * 32),
            ),
            final_depth_conv=self._create_conv_config(
                self.input_height, self.input_width, 32, 1,
                self.parameters["decoder"]["final"],
                kernel_size=(1, 1), padding=(0, 0),
                activation=None, # No activation for final depth output
                sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=6 * 32),
            ),

            # Uncertainty head
            uncertainty_conv1=self._create_conv_config(
                self.input_height, self.input_width, 1, 16,
                self.parameters["uncertainty"]["conv1"],
                sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=6 * 32),
            ),
            uncertainty_conv2=self._create_conv_config(
                self.input_height, self.input_width, 16, 1,
                self.parameters["uncertainty"]["conv2"],
                kernel_size=(1, 1), padding=(0, 0),
                activation=None, # Will apply softplus separately
                sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=6 * 32),
            ),

            timestep_embed_dim=256,
            num_inference_steps=20,
        )


def create_monodiffusion_configs_from_parameters(
    parameters: Dict,
    batch_size: int = 1,
    input_height: int = 192,
    input_width: int = 640,
    in_channels: int = 3,
) -> TtMonoDiffusionLayerConfigs:
    """
    Create MonoDiffusion configuration from preprocessed parameters
    Following vanilla_unet pattern
    """
    builder = TtMonoDiffusionConfigBuilder(
        parameters=parameters,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        in_channels=in_channels,
    )
    return builder.build_configs()

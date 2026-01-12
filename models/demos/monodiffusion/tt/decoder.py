# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Multi-scale decoder for MonoDiffusion
Following vanilla_unet decoder pattern with transpose convolutions
"""

import ttnn
from typing import List
from models.demos.monodiffusion.tt.config import TtMonoDiffusionLayerConfigs
from models.demos.monodiffusion.tt.common import concatenate_skip_connection
from models.tt_cnn.tt.builder import TtConv2d


def transpose_conv2d(
    input_tensor: ttnn.Tensor,
    in_channels: int,
    out_channels: int,
    input_height: int,
    input_width: int,
    batch_size: int,
    device: ttnn.Device,
    act_block_h_override: int = 32,
    fp32_dest_acc_en: bool = True,
    packer_l1_acc: bool = True,
) -> ttnn.Tensor:
    """
    Transpose convolution for upsampling
    Following vanilla_unet pattern
    """
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation=True,
        enable_act_double_buffer=False,
        output_layout=ttnn.TILE_LAYOUT,
        act_block_h_override=act_block_h_override,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )

    # For now, use simple upsampling + convolution
    # TODO: Implement proper transpose convolution when weights are available
    # Reshape for upsampling
    x_reshaped = ttnn.reshape(input_tensor, (batch_size, input_height, input_width, in_channels))

    # Upsample 2x
    x_upsampled = ttnn.upsample(x_reshaped, (2, 2), memory_config=input_tensor.memory_config())

    # Reshape back to (1, 1, N, C) format
    output = ttnn.reshape(
        x_upsampled,
        (1, 1, batch_size * input_height * 2 * input_width * 2, in_channels)
    )

    return output


class TtMonoDiffusionDecoder:
    """
    Multi-scale decoder for depth map refinement
    Follows vanilla_unet decoder pattern
    """

    def __init__(self, configs: TtMonoDiffusionLayerConfigs, device: ttnn.Device):
        self.device = device
        self.configs = configs

        # Build decoder convolution layers using TtConv2d from builder
        self.conv1 = TtConv2d(configs.decoder_conv1, device)
        self.conv2 = TtConv2d(configs.decoder_conv2, device)
        self.conv3 = TtConv2d(configs.decoder_conv3, device)
        self.conv4 = TtConv2d(configs.decoder_conv4, device)

        # Final depth prediction layer
        self.final_conv = TtConv2d(configs.final_depth_conv, device)

    def __call__(
        self,
        coarse_depth: ttnn.Tensor,
        encoder_features: List[ttnn.Tensor]
    ) -> ttnn.Tensor:
        """
        Forward pass through decoder

        Args:
            coarse_depth: Coarse depth map from diffusion U-Net
            encoder_features: Multi-scale features from encoder for skip connections

        Returns:
            Refined depth map at original input resolution
        """
        x = coarse_depth

        # Decoder block 1: upsample + conv1
        x = transpose_conv2d(
            x,
            in_channels=512,
            out_channels=512,
            input_height=self.configs.decoder_conv1.input_height // 2,
            input_width=self.configs.decoder_conv1.input_width // 2,
            batch_size=self.configs.decoder_conv1.batch_size,
            device=self.device,
            act_block_h_override=2 * 32,
        )
        # Skip connection from encoder if available
        if len(encoder_features) > 3:
            x = concatenate_skip_connection(x, encoder_features[3], use_row_major_layout=True)
        x = self.conv1(x)

        # Decoder block 2: upsample + conv2
        x = transpose_conv2d(
            x,
            in_channels=256,
            out_channels=256,
            input_height=self.configs.decoder_conv2.input_height // 2,
            input_width=self.configs.decoder_conv2.input_width // 2,
            batch_size=self.configs.decoder_conv2.batch_size,
            device=self.device,
            act_block_h_override=3 * 32,
        )
        if len(encoder_features) > 2:
            x = concatenate_skip_connection(x, encoder_features[2], use_row_major_layout=True)
        x = self.conv2(x)

        # Decoder block 3: upsample + conv3
        x = transpose_conv2d(
            x,
            in_channels=128,
            out_channels=128,
            input_height=self.configs.decoder_conv3.input_height // 2,
            input_width=self.configs.decoder_conv3.input_width // 2,
            batch_size=self.configs.decoder_conv3.batch_size,
            device=self.device,
            act_block_h_override=5 * 32,
        )
        if len(encoder_features) > 1:
            x = concatenate_skip_connection(x, encoder_features[1], use_row_major_layout=True)
        x = self.conv3(x)

        # Decoder block 4: upsample + conv4
        x = transpose_conv2d(
            x,
            in_channels=64,
            out_channels=64,
            input_height=self.configs.decoder_conv4.input_height // 2,
            input_width=self.configs.decoder_conv4.input_width // 2,
            batch_size=self.configs.decoder_conv4.batch_size,
            device=self.device,
            act_block_h_override=10 * 32,
        )
        if len(encoder_features) > 0:
            x = concatenate_skip_connection(x, encoder_features[0], use_row_major_layout=False)
        x = self.conv4(x)

        # Final depth prediction
        depth_map = self.final_conv(x)

        # Apply sigmoid to constrain depth values [0, 1]
        depth_map = ttnn.sigmoid(depth_map, memory_config=depth_map.memory_config())

        return depth_map

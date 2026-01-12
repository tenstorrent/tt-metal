# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Diffusion U-Net for MonoDiffusion
Conditional denoising network with timestep embedding
Following vanilla_unet U-Net pattern
"""

import ttnn
from typing import List, Optional
from models.demos.monodiffusion.tt.config import TtMonoDiffusionLayerConfigs
from models.demos.monodiffusion.tt.common import concatenate_skip_connection
from models.tt_cnn.tt.builder import TtConv2d


class TtResidualBlock:
    """
    Residual block with timestep conditioning for diffusion U-Net
    Similar to vanilla_unet encoder/decoder blocks
    """

    def __init__(self, conv1: TtConv2d, conv2: TtConv2d):
        self.conv1 = conv1
        self.conv2 = conv2

    def __call__(self, x: ttnn.Tensor, time_emb: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        """
        Forward pass with optional timestep conditioning
        """
        residual = x

        # First convolution
        x = self.conv1(x)

        # TODO: Add timestep embedding projection and addition
        # if time_emb is not None:
        #     time_proj = project_timestep(time_emb)
        #     x = ttnn.add(x, time_proj)

        # Second convolution
        x = self.conv2(x)

        # Residual connection
        x = ttnn.add(x, residual, memory_config=x.memory_config())

        return x


class TtDiffusionUNet:
    """
    Diffusion U-Net for depth map denoising
    Follows vanilla_unet architecture pattern
    """

    def __init__(self, configs: TtMonoDiffusionLayerConfigs, device: ttnn.Device):
        self.device = device
        self.configs = configs

        # Build U-Net layers using TtConv2d from builder
        # Down blocks
        self.down1_conv1 = TtConv2d(configs.unet_down1_conv1, device)
        self.down1_conv2 = TtConv2d(configs.unet_down1_conv2, device)

        # Middle blocks
        self.mid_conv1 = TtConv2d(configs.unet_mid_conv1, device)
        self.mid_conv2 = TtConv2d(configs.unet_mid_conv2, device)

        # Up blocks
        self.up1_conv1 = TtConv2d(configs.unet_up1_conv1, device)
        self.up1_conv2 = TtConv2d(configs.unet_up1_conv2, device)

    def downsample(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Downsample using stride-2 convolution or pooling
        For now, use simple reshaping
        """
        # TODO: Implement proper downsampling
        return x

    def upsample(self, x: ttnn.Tensor, target_height: int, target_width: int) -> ttnn.Tensor:
        """
        Upsample using transpose convolution or interpolation
        Following vanilla_unet pattern
        """
        batch_size = self.configs.unet_up1_conv1.batch_size
        channels = x.shape[-1]

        # Reshape to (B, H, W, C)
        current_height = target_height // 2
        current_width = target_width // 2
        x_reshaped = ttnn.reshape(x, (batch_size, current_height, current_width, channels))

        # Upsample 2x
        x_upsampled = ttnn.upsample(x_reshaped, (2, 2), memory_config=x.memory_config())

        # Reshape back to (1, 1, N, C)
        x_upsampled = ttnn.reshape(
            x_upsampled,
            (1, 1, batch_size * target_height * target_width, channels)
        )

        return x_upsampled

    def __call__(
        self,
        x: ttnn.Tensor,
        timestep_emb: Optional[ttnn.Tensor] = None,
        encoder_features: Optional[List[ttnn.Tensor]] = None
    ) -> ttnn.Tensor:
        """
        Forward pass through diffusion U-Net

        Args:
            x: Noisy depth map
            timestep_emb: Timestep embedding for conditioning
            encoder_features: Multi-scale features from encoder (for conditioning)

        Returns:
            Denoised depth map (or noise prediction)
        """
        # Downsampling path
        x = self.down1_conv1(x)
        skip1 = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = self.down1_conv2(x)

        # Downsample
        x = self.downsample(x)

        # Middle blocks
        x = self.mid_conv1(x)
        x = self.mid_conv2(x)

        # Upsampling path
        x = self.upsample(
            x,
            target_height=self.configs.unet_up1_conv1.input_height,
            target_width=self.configs.unet_up1_conv1.input_width
        )

        # Concatenate with skip connection
        x = concatenate_skip_connection(x, skip1, use_row_major_layout=True)

        x = self.up1_conv1(x)
        x = self.up1_conv2(x)

        return x

    def denoise_step(
        self,
        noisy_depth: ttnn.Tensor,
        timestep: int,
        timestep_emb: Optional[ttnn.Tensor] = None,
        encoder_features: Optional[List[ttnn.Tensor]] = None
    ) -> ttnn.Tensor:
        """
        Single denoising step in the diffusion process
        Simplified DDPM update rule
        """
        # Predict noise
        noise_pred = self(noisy_depth, timestep_emb, encoder_features)

        # Update depth map (simplified DDPM update)
        # In full implementation, this would use proper diffusion scheduler
        alpha_t = 1.0 - (timestep / 1000.0)

        # Compute: x_t-1 = x_t - (1 - alpha_t) * noise_pred
        scaled_noise = ttnn.multiply(noise_pred, 1.0 - alpha_t, memory_config=noise_pred.memory_config())
        denoised = ttnn.subtract(
            noisy_depth,
            scaled_noise,
            memory_config=noisy_depth.memory_config()
        )

        ttnn.deallocate(scaled_noise)

        return denoised

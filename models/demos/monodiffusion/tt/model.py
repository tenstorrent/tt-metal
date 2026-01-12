# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Main MonoDiffusion model for monocular depth estimation
Following vanilla_unet model pattern
"""

import ttnn
import torch
from typing import Tuple, List, Optional

from models.demos.monodiffusion.tt.config import (
    TtMonoDiffusionLayerConfigs,
    create_monodiffusion_configs_from_parameters,
)
from models.demos.monodiffusion.tt.encoder import TtMonoDiffusionEncoder
from models.demos.monodiffusion.tt.diffusion_unet import TtDiffusionUNet
from models.demos.monodiffusion.tt.decoder import TtMonoDiffusionDecoder
from models.demos.monodiffusion.tt.uncertainty_head import TtUncertaintyHead
from models.demos.monodiffusion.tt.timestep_embedding import TtTimestepEmbedding


class TtMonoDiffusion:
    """
    Main MonoDiffusion model for monocular depth estimation
    Following vanilla_unet architecture pattern

    Architecture:
    1. Encoder: Extract multi-scale features from input image
    2. Diffusion U-Net: Iterative denoising to generate depth map
    3. Decoder: Multi-scale refinement of depth map
    4. Uncertainty Head: Estimate per-pixel uncertainty
    """

    def __init__(self, configs: TtMonoDiffusionLayerConfigs, device: ttnn.Device):
        self.device = device
        self.configs = configs

        # Initialize model components
        self.encoder = TtMonoDiffusionEncoder(configs, device)
        self.diffusion_unet = TtDiffusionUNet(configs, device)
        self.decoder = TtMonoDiffusionDecoder(configs, device)
        self.uncertainty_head = TtUncertaintyHead(configs, device)
        self.timestep_embedding = TtTimestepEmbedding(configs.timestep_embed_dim, device)

        # Diffusion parameters
        self.num_inference_steps = configs.num_inference_steps

    def preprocess_input_tensor(self, x: ttnn.Tensor, deallocate_input: bool = True) -> ttnn.Tensor:
        """
        Preprocess input tensor to HWC format
        Following vanilla_unet pattern
        """
        output = ttnn.experimental.convert_to_hwc(x)
        if deallocate_input:
            ttnn.deallocate(x)
        return output

    def diffusion_process(
        self,
        encoder_features: List[ttnn.Tensor],
        num_steps: Optional[int] = None
    ) -> ttnn.Tensor:
        """
        Run diffusion process to generate depth map
        Simplified implementation for initial bring-up

        Args:
            encoder_features: Multi-scale features from encoder
            num_steps: Number of denoising steps (default: self.num_inference_steps)

        Returns:
            Generated depth map
        """
        if num_steps is None:
            num_steps = self.num_inference_steps

        # Start with encoder features as initialization
        depth_map = encoder_features[-1]

        # Iterative denoising
        for t in range(num_steps - 1, -1, -1):
            # Create timestep tensor
            timestep = torch.tensor([t], dtype=torch.long)

            # Get timestep embedding
            timestep_emb = self.timestep_embedding(timestep)

            # Denoise step
            depth_map = self.diffusion_unet.denoise_step(
                depth_map,
                t,
                timestep_emb,
                encoder_features
            )

        return depth_map

    def __call__(
        self,
        input_tensor: ttnn.Tensor,
        return_uncertainty: bool = True,
        num_inference_steps: Optional[int] = None,
        deallocate_input_activation: bool = True
    ) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor]]:
        """
        Forward pass through MonoDiffusion model
        Following vanilla_unet __call__ pattern

        Args:
            input_tensor: Input RGB image (B, C, H, W)
            return_uncertainty: Whether to compute uncertainty map
            num_inference_steps: Number of diffusion steps (default: config value)
            deallocate_input_activation: Whether to deallocate input tensor

        Returns:
            - depth_map: Predicted depth map
            - uncertainty_map: Uncertainty map (if return_uncertainty=True)
        """
        # Preprocess input
        input_tensor = self.preprocess_input_tensor(input_tensor, deallocate_input_activation)

        # 1. Feature extraction
        encoded_features, multi_scale_features = self.encoder(input_tensor)

        # 2. Diffusion process to generate coarse depth
        coarse_depth = self.diffusion_process(
            multi_scale_features,
            num_steps=num_inference_steps
        )

        # 3. Multi-scale refinement
        refined_depth = self.decoder(coarse_depth, multi_scale_features)

        # 4. Uncertainty estimation
        uncertainty_map = None
        if return_uncertainty:
            uncertainty_map = self.uncertainty_head(refined_depth, multi_scale_features)

        return refined_depth, uncertainty_map


def create_monodiffusion_from_configs(
    configs: TtMonoDiffusionLayerConfigs,
    device: ttnn.Device
) -> TtMonoDiffusion:
    """
    Factory function to create MonoDiffusion model from configs
    Following vanilla_unet pattern
    """
    return TtMonoDiffusion(configs, device)


def create_monodiffusion_from_parameters(
    parameters: dict,
    device: ttnn.Device,
    batch_size: int = 1,
    input_height: int = 192,
    input_width: int = 640,
) -> TtMonoDiffusion:
    """
    Create MonoDiffusion model from preprocessed parameters
    Following vanilla_unet pattern

    Args:
        parameters: Preprocessed weight parameters
        device: TT device
        batch_size: Batch size
        input_height: Input image height
        input_width: Input image width

    Returns:
        Initialized MonoDiffusion model
    """
    configs = create_monodiffusion_configs_from_parameters(
        parameters=parameters,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
    )

    return create_monodiffusion_from_configs(configs, device)

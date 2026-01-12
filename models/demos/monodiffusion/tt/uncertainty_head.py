# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Uncertainty estimation head for MonoDiffusion
Following vanilla_unet pattern with TtConv2d
"""

import ttnn
from typing import List, Optional
from models.demos.monodiffusion.tt.config import TtMonoDiffusionLayerConfigs
from models.tt_cnn.tt.builder import TtConv2d


class TtUncertaintyHead:
    """
    Uncertainty estimation head for MonoDiffusion
    Predicts per-pixel uncertainty/confidence for depth predictions
    """

    def __init__(self, configs: TtMonoDiffusionLayerConfigs, device: ttnn.Device):
        self.device = device
        self.configs = configs

        # Build uncertainty head layers using TtConv2d from builder
        self.conv1 = TtConv2d(configs.uncertainty_conv1, device)
        self.conv2 = TtConv2d(configs.uncertainty_conv2, device)

    def __call__(
        self,
        depth_map: ttnn.Tensor,
        encoder_features: Optional[List[ttnn.Tensor]] = None
    ) -> ttnn.Tensor:
        """
        Predict uncertainty map for depth predictions

        Args:
            depth_map: Predicted depth map
            encoder_features: Optional encoder features for context (unused for now)

        Returns:
            Uncertainty map (same spatial dimensions as depth map)
        """
        x = depth_map

        # Process through convolution layers
        x = self.conv1(x)
        uncertainty = self.conv2(x)

        # Apply softplus to ensure positive uncertainty values
        # softplus(x) = log(1 + exp(x))
        uncertainty = ttnn.softplus(uncertainty, memory_config=uncertainty.memory_config())

        return uncertainty

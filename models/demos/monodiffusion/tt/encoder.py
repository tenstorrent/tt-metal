# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Feature encoder for MonoDiffusion
ResNet-like architecture following vanilla_unet encoder pattern
"""

import ttnn
from typing import List, Tuple
from models.demos.monodiffusion.tt.config import TtMonoDiffusionLayerConfigs
from models.tt_cnn.tt.builder import TtConv2d, TtMaxPool2d


class TtMonoDiffusionEncoder:
    """
    Feature encoder for MonoDiffusion
    Follows vanilla_unet encoder pattern with TtConv2d and TtMaxPool2d
    """

    def __init__(self, configs: TtMonoDiffusionLayerConfigs, device: ttnn.Device):
        self.device = device
        self.configs = configs

        # Build encoder layers using TtConv2d and TtMaxPool2d from builder
        self.conv1 = TtConv2d(configs.encoder_conv1, device)
        self.conv2 = TtConv2d(configs.encoder_conv2, device)
        self.conv3 = TtConv2d(configs.encoder_conv3, device)
        self.conv4 = TtConv2d(configs.encoder_conv4, device)

        self.pool1 = TtMaxPool2d(configs.encoder_pool1, device)
        self.pool2 = TtMaxPool2d(configs.encoder_pool2, device)

    def __call__(self, x: ttnn.Tensor) -> Tuple[ttnn.Tensor, List[ttnn.Tensor]]:
        """
        Forward pass through encoder

        Args:
            x: Input tensor in HWC format (batch, height, width, channels)

        Returns:
            - Final encoded features
            - List of multi-scale features for skip connections
        """
        multi_scale_features = []

        # Encoder block 1: conv1 -> pool1
        x = self.conv1(x)
        multi_scale_features.append(ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG))
        x = self.pool1(x)

        # Encoder block 2: conv2 -> pool2
        x = self.conv2(x)
        multi_scale_features.append(ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG))
        x = self.pool2(x)

        # Encoder block 3: conv3 (no pooling)
        x = self.conv3(x)
        multi_scale_features.append(ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG))

        # Encoder block 4: conv4 (no pooling)
        x = self.conv4(x)
        multi_scale_features.append(ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG))

        return x, multi_scale_features

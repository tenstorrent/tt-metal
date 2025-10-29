# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.common.lightweightmodule import LightweightModule
from .tt_config import get_resnet_config
from .components.tt_components import (
    ResNetTimeEmbedding,
    ResNetShortcutConnection,
)
from .components.weight_loader import WeightLoader
from .components.convolution_layer import ConvolutionLayer
from .components.group_normalization_layer import GroupNormalizationLayer


class TtResnetBlock2D(LightweightModule):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
    ):
        super().__init__()

        self.device = device
        self.module_path = module_path

        # Configuration setup
        self.block_config = get_resnet_config(module_path)
        self.use_conv_shortcut = self.block_config.use_conv_shortcut

        # Initialize components
        self._initialize_components(state_dict)

    def _initialize_components(self, state_dict):
        # Order of layers:
        # 1. group_norm_1
        # 2. conv2d_1
        # 3. time_embedding
        # 4. group_norm_2
        # 5. conv2d_2
        # 6. shortcut (if use_conv_shortcut is True)

        self.weight_loader = WeightLoader(self, state_dict, self.module_path, use_conv_shortcut=self.use_conv_shortcut)

        self.norm_layer_1 = GroupNormalizationLayer(
            self.device,
            self.weight_loader.norm_weights_1,
            self.weight_loader.norm_bias_1,
            self.block_config.norm1,
        )

        self.norm_layer_2 = GroupNormalizationLayer(
            self.device,
            self.weight_loader.norm_weights_2,
            self.weight_loader.norm_bias_2,
            self.block_config.norm2,
        )

        self.conv_layer_1 = ConvolutionLayer(
            self.device,
            self.weight_loader.conv_weights_1,
            self.weight_loader.conv_bias_1,
            self.block_config.conv1,
        )

        self.conv_layer_2 = ConvolutionLayer(
            self.device,
            self.weight_loader.conv_weights_2,
            self.weight_loader.conv_bias_2,
            self.block_config.conv2,
        )

        self.time_embedding = ResNetTimeEmbedding(
            self.device,
            self.weight_loader.time_emb_weights,
            self.weight_loader.time_emb_bias,
        )

        self.shortcut_connection = ResNetShortcutConnection(
            self.device,
            use_conv_shortcut=self.use_conv_shortcut,
            conv_weights=self.weight_loader.conv_weights_3,
            conv_bias=self.weight_loader.conv_bias_3,
            conv_config=self.block_config.conv_shortcut if self.use_conv_shortcut else None,
        )

    def forward(self, input_tensor, temb, input_shape):
        B, C, H, W = input_shape
        hidden_states = input_tensor

        # First normalization + activation
        hidden_states = self.norm_layer_1.forward(hidden_states, B, C, H, W)
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        hidden_states = ttnn.silu(hidden_states)

        # First convolution
        hidden_states, [C, H, W] = self.conv_layer_1.forward(hidden_states, B, C, H, W)

        # Add time embedding
        hidden_states = self.time_embedding.forward(hidden_states, temb)

        # Second normalization + activation
        hidden_states = self.norm_layer_2.forward(hidden_states, B, C, H, W)
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        hidden_states = ttnn.silu(hidden_states)

        # Second convolution
        hidden_states, [C, H, W] = self.conv_layer_2.forward(hidden_states, B, C, H, W)

        # Apply shortcut connection
        hidden_states, [C, H, W] = self.shortcut_connection.forward(input_tensor, hidden_states, input_shape)

        return hidden_states, [C, H, W]

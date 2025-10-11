# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_refiner.tt.components.convolution_layer import ConvolutionLayer
from models.experimental.stable_diffusion_xl_refiner.tt.tt_config import get_upsample_config
from .components.weight_loader import WeightLoader


class TtUpsample2D(LightweightModule):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
    ):
        super().__init__()

        self.device = device
        self.module_path = module_path

        self.scale_factor = 2

        # Configuration setup
        self.block_config = get_upsample_config(module_path)

        # Initialize components
        self._initialize_components(state_dict)

    def _initialize_components(self, state_dict):
        # Order of layers:
        # 1. conv_layer

        self.weight_loader = WeightLoader(self, state_dict, self.module_path)

        self.conv_layer = ConvolutionLayer(
            self.device,
            self.weight_loader.conv_weight,
            self.weight_loader.conv_bias,
            self.block_config.conv,
        )

    def forward(self, input_tensor):
        hidden_states = input_tensor

        # TILE_LAYOUT Fails on one of the cases bcs 16x16 is not tile alligned
        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.upsample(hidden_states, (self.scale_factor, self.scale_factor))
        B, H, W, C = list(hidden_states.shape)
        hidden_states, [C, H, W] = self.conv_layer.forward(hidden_states, B, C, H, W)
        return hidden_states, [C, H, W]

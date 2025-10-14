# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_refiner.tt.components.convolution_layer import ConvolutionLayer
from models.experimental.stable_diffusion_xl_refiner.tt.tt_config import get_downsample_config
from .components.weight_loader import WeightLoader


class TtDownsample2D(LightweightModule):
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
        self.block_config = get_downsample_config(module_path)

        # Initialize components
        self._initialize_components(state_dict)

    def _initialize_components(self, state_dict):
        # Order of layers:
        # 1. conv_layer

        # Load weights
        self.weight_loader = WeightLoader(self, state_dict, self.module_path)

        self.conv_layer = ConvolutionLayer(
            self.device,
            self.weight_loader.conv_weight,
            self.weight_loader.conv_bias,
            self.block_config.conv,
        )

    def forward(self, hidden_states, input_shape):
        B, C, H, W = input_shape

        hidden_states, [C, H, W] = self.conv_layer.forward(hidden_states, B, C, H, W)

        return hidden_states, [C, H, W]

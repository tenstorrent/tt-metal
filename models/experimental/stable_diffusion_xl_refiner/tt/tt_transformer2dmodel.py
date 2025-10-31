# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_refiner.tt.components.group_normalization_layer import (
    GroupNormalizationLayer,
)
from models.experimental.stable_diffusion_xl_refiner.tt.tt_transformerblock import TtBasicTransformerBlock
from models.experimental.stable_diffusion_xl_refiner.tt.tt_config import get_transformer2dmodel_config
from .components.weight_loader import WeightLoader


class TtTransformer2DModel(LightweightModule):
    def __init__(self, device, state_dict, module_path):
        super().__init__()

        self.device = device
        self.module_path = module_path

        # Configuration setup
        self.block_config = get_transformer2dmodel_config(module_path)

        self.compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self._initialize_components(state_dict)

    def _initialize_components(self, state_dict):
        # Order of layers:
        # 1. group_norm
        # 2. input_projection
        # 3. transformer_blocks (list of 4 blocks)
        # 5. output_projection

        # Load weights
        self.weight_loader = WeightLoader(self, state_dict, self.module_path)

        self.norm_layer = GroupNormalizationLayer(
            self.device,
            self.weight_loader.norm_weight,
            self.weight_loader.norm_bias,
            self.block_config.norm,
        )

        self.weight_loader.prepare_linear_params(ttnn.bfloat16)

        self.transformer_blocks = []
        for i in range(self.block_config.num_transformer_blocks):
            transformer_block_module_path = f"{self.module_path}.transformer_blocks.{i}"

            self.transformer_blocks.append(
                TtBasicTransformerBlock(
                    device=self.device,
                    state_dict=state_dict,
                    module_path=transformer_block_module_path,
                )
            )

    def forward(self, input_tensor, input_shape, encoder_hidden_states=None):
        B, C, H, W = input_shape
        hidden_states = input_tensor

        # GroupNorm
        hidden_states = self.norm_layer.forward(hidden_states, B, C, H, W)
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

        # Input projection
        hidden_states = ttnn.linear(
            hidden_states,
            self.weight_loader.weights_in,
            bias=self.weight_loader.bias_in,
            compute_kernel_config=self.compute_config,
        )

        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block.forward(hidden_states, encoder_hidden_states)

        # Output projection
        hidden_states = ttnn.linear(
            hidden_states,
            self.weight_loader.weights_out,
            bias=self.weight_loader.bias_out,
            compute_kernel_config=self.compute_config,
        )

        hidden_states = ttnn.add(hidden_states, input_tensor, use_legacy=False)

        return hidden_states

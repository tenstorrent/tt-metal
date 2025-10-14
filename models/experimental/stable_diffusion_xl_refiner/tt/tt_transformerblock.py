# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_refiner.tt.tt_attention import TtAttention
from models.experimental.stable_diffusion_xl_refiner.tt.components.tt_components import TransformerBlockLayerNorm
from models.experimental.stable_diffusion_xl_refiner.tt.tt_feedforward import TtFeedForward
from models.experimental.stable_diffusion_xl_refiner.tt.tt_config import get_transformerblock_config
from .components.weight_loader import WeightLoader


class TtBasicTransformerBlock(LightweightModule):
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
        self.block_config = get_transformerblock_config(module_path)

        self._initialize_components(state_dict)

    def _initialize_components(self, state_dict):
        # Order of layers:
        # 1. layer_norm_1
        # 2. attention_1
        # 3. layer_norm_2
        # 4. attention_2
        # 5. layer_norm_3
        # 6. feed_forward

        self.weight_loader = WeightLoader(self, state_dict, self.module_path)

        self.layer_norm_1 = TransformerBlockLayerNorm(
            self.device,
            self.weight_loader.norm_weights_1,
            self.weight_loader.norm_bias_1,
        )

        self.attention_1 = TtAttention(
            device=self.device,
            state_dict=state_dict,
            module_path=f"{self.module_path}.attn1",
            num_attn_heads=self.block_config.num_attn_heads,
        )

        self.layer_norm_2 = TransformerBlockLayerNorm(
            self.device,
            self.weight_loader.norm_weights_2,
            self.weight_loader.norm_bias_2,
        )

        self.attention_2 = TtAttention(
            device=self.device,
            state_dict=state_dict,
            module_path=f"{self.module_path}.attn2",
            num_attn_heads=self.block_config.num_attn_heads,
        )

        self.layer_norm_3 = TransformerBlockLayerNorm(
            self.device,
            self.weight_loader.norm_weights_3,
            self.weight_loader.norm_bias_3,
        )

        self.feedforward = TtFeedForward(
            device=self.device,
            state_dict=state_dict,
            module_path=f"{self.module_path}.ff",
        )

    def forward(self, input_tensor, encoder_tensor):
        hidden_states = input_tensor

        # Self-attention layer
        hidden_states = self.layer_norm_1.forward(hidden_states)
        hidden_states = self.attention_1.forward(hidden_states, None)
        attn_result = ttnn.add(input_tensor, hidden_states, use_legacy=False)

        ttnn.deallocate(input_tensor)

        # Cross-attention layer
        hidden_states = self.layer_norm_2.forward(attn_result)
        hidden_states = self.attention_2.forward(hidden_states, encoder_tensor)
        attn_result = ttnn.add(attn_result, hidden_states, use_legacy=False)

        # Feedforward layer
        hidden_states = self.layer_norm_3.forward(attn_result)
        hidden_states = self.feedforward.forward(hidden_states)
        attn_result = ttnn.add(attn_result, hidden_states, use_legacy=False)

        return attn_result

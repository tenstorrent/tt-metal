# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_refiner.tt.tt_geglu import TtGEGLU
from .components.weight_loader import WeightLoader


class TtFeedForward(LightweightModule):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
    ):
        super().__init__()

        self.device = device

        self.weight_loader = WeightLoader(self, state_dict, module_path)

        # Prepare linear parameters for the final linear layer
        self.weight_loader.prepare_linear_params(ttnn.bfloat16)

        self.tt_geglu = TtGEGLU(device, state_dict, f"{module_path}.net.0")

    def forward(self, hidden_states):
        hidden_states = self.tt_geglu.forward(hidden_states)

        hidden_states = ttnn.linear(
            hidden_states,
            self.weight_loader.linear_weights,
            bias=self.weight_loader.linear_bias,
        )
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)

        return hidden_states

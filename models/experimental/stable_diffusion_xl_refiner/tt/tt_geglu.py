# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.common.lightweightmodule import LightweightModule
from .components.weight_loader import WeightLoader


class TtGEGLU(LightweightModule):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
    ):
        super().__init__()

        self.device = device
        self.module_path = module_path

        self.weight_loader = WeightLoader(self, state_dict, module_path)

        # Prepare linear parameters for both value and gate projections
        self.weight_loader.prepare_linear_params(ttnn.bfloat16)

    def forward(self, input_tensor):
        hidden_states = ttnn.linear(
            input_tensor,
            self.weight_loader.value_weights,
            bias=self.weight_loader.value_bias,
        )

        gate = ttnn.linear(
            input_tensor,
            self.weight_loader.gate_weights,
            bias=self.weight_loader.gate_bias,
        )

        gate = ttnn.gelu(gate)

        hidden_states = ttnn.mul_(hidden_states, gate, use_legacy=False)
        return hidden_states

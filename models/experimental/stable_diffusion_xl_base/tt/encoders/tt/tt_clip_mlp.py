# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn

from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import prepare_linear_params


class TtClipMLP(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        model_config,
    ):
        super().__init__()

        self.device = device

        fc1_weights = state_dict[f"{module_path}.fc1.weight"].unsqueeze(0).unsqueeze(0)
        fc1_bias = state_dict[f"{module_path}.fc1.bias"]
        fc2_weights = state_dict[f"{module_path}.fc2.weight"].unsqueeze(0).unsqueeze(0)
        fc2_bias = state_dict[f"{module_path}.fc2.bias"]

        self.tt_fc1_weights, self.tt_fc1_bias = prepare_linear_params(
            device, fc1_weights, fc1_bias, model_config.ff_weights_dtype
        )
        self.tt_fc2_weights, self.tt_fc2_bias = prepare_linear_params(
            device, fc2_weights, fc2_bias, model_config.ff_weights_dtype
        )

        self.default_compute_kernel_config = model_config.get_mm_compute_config(module_path)

    def forward(self, hidden_states):
        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_fc1_weights,
            bias=self.tt_fc1_bias,
            compute_kernel_config=self.default_compute_kernel_config,
            activation="gelu",
        )

        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_fc2_weights,
            bias=self.tt_fc2_bias,
            compute_kernel_config=self.default_compute_kernel_config,
        )

        return hidden_states

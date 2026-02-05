# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import prepare_linear_params


class TtTimestepEmbedding(LightweightModule):
    def __init__(self, device, state_dict, module_path, model_config, linear_weights_dtype=ttnn.bfloat16):
        super().__init__()

        self.device = device
        weights_1 = state_dict[f"{module_path}.linear_1.weight"]
        bias_1 = state_dict[f"{module_path}.linear_1.bias"]

        weights_2 = state_dict[f"{module_path}.linear_2.weight"]
        bias_2 = state_dict[f"{module_path}.linear_2.bias"]

        self.tt_weights_1, self.tt_bias_1 = prepare_linear_params(device, weights_1, bias_1, linear_weights_dtype)
        self.tt_weights_2, self.tt_bias_2 = prepare_linear_params(device, weights_2, bias_2, linear_weights_dtype)

        self.linear_1_program_config = model_config.get_matmul_config(f"{module_path}.linear_1")
        self.linear_2_program_config = model_config.get_matmul_config(f"{module_path}.linear_2")
        assert self.linear_1_program_config is not None
        assert self.linear_2_program_config is not None

    def forward(self, sample):
        sample = ttnn.linear(
            sample,
            self.tt_weights_1,
            bias=self.tt_bias_1,
            program_config=self.linear_1_program_config if self.linear_1_program_config else None,
            activation="silu" if not self.linear_1_program_config else None,
        )
        sample = ttnn.linear(
            sample,
            self.tt_weights_2,
            bias=self.tt_bias_2,
            program_config=self.linear_2_program_config if self.linear_2_program_config else None,
            activation="silu" if not self.linear_2_program_config else None,
        )
        return sample

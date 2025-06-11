# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn

from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import prepare_linear_params


class TtTimestepEmbedding(nn.Module):
    def __init__(self, device, state_dict, module_path, linear_weights_dtype=ttnn.bfloat16):
        super().__init__()

        self.device = device
        weights_1 = state_dict[f"{module_path}.linear_1.weight"].unsqueeze(0).unsqueeze(0)
        bias_1 = state_dict[f"{module_path}.linear_1.bias"]

        weights_2 = state_dict[f"{module_path}.linear_2.weight"].unsqueeze(0).unsqueeze(0)
        bias_2 = state_dict[f"{module_path}.linear_2.bias"]

        self.tt_weights_1, self.tt_bias_1 = prepare_linear_params(device, weights_1, bias_1, linear_weights_dtype)
        self.tt_weights_2, self.tt_bias_2 = prepare_linear_params(device, weights_2, bias_2, linear_weights_dtype)

    def forward(self, sample):
        sample = ttnn.linear(sample, self.tt_weights_1, bias=self.tt_bias_1, activation="silu")
        sample = ttnn.linear(
            sample,
            self.tt_weights_2,
            bias=self.tt_bias_2,
        )
        return sample

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from models.utility_functions import pad_by_zero

import ttnn


class TtT5LayerNorm(torch.nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.variance_epsilon = config["layer_norm_epsilon"]
        self.device = device

        # get weights
        pytorch_weights = state_dict[f"{base_address}.weight"]

        self.weight = pad_by_zero(pytorch_weights, device)[0]

    def forward(self, hidden_states):
        return ttnn.rms_norm(hidden_states, epsilon=self.variance_epsilon, weight=self.weight)

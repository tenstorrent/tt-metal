# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
from models.utility_functions import pad_by_zero


class TtT5LayerNorm(torch.nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()
        self.variance_epsilon = config.layer_norm_epsilon
        self.device = device

        layer_norm_weights = state_dict[f"{base_address}.weight"]

        self.weight = pad_by_zero(layer_norm_weights, device)[0]

    def forward(self, hidden_states: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        result = tt_lib.tensor.rmsnorm(
            hidden_states,
            self.variance_epsilon,
            self.weight,
        )
        return result

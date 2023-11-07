# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
from models.helper_funcs import Linear

from models.utility_functions import (
    torch_to_tt_tensor_rm,
)


class TtMLP(torch.nn.Module):
    def __init__(self, base_address, config, state_dict, device):
        super().__init__()
        # Get the weights
        self.tt_weight_c_fc = state_dict[f"{base_address}.c_fc.weight"]
        self.tt_weight_c_proj = state_dict[f"{base_address}.c_proj.weight"]
        self.config = config
        self.device = device

        # Push weights to Tt device
        self.tt_weight_c_fc = torch_to_tt_tensor_rm(self.tt_weight_c_fc, self.device)

        self.tt_weight_c_proj = torch_to_tt_tensor_rm(self.tt_weight_c_proj, self.device)

        # Load biases
        self.tt_bias_c_fc = torch_to_tt_tensor_rm(state_dict[f"{base_address}.c_fc.bias"], self.device)

        self.tt_bias_c_proj = torch_to_tt_tensor_rm(state_dict[f"{base_address}.c_proj.bias"], self.device)

        self.tt_weight_c_fc = tt_lib.tensor.transpose(self.tt_weight_c_fc, -2, -1)
        self.tt_weight_c_proj = tt_lib.tensor.transpose(self.tt_weight_c_proj, -2, -1)

        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, self.tt_weight_c_fc, self.tt_bias_c_fc)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, self.tt_weight_c_proj, self.tt_bias_c_proj)

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        x1 = self.c_fc(x)
        x2 = tt_lib.tensor.gelu(x1)
        x3 = self.c_proj(x2)

        return x3

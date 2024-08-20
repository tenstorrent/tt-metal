# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn


from models.utility_functions import (
    torch_to_tt_tensor_rm,
)
from models.experimental.swin.swin_helper_funcs import linear as TtLinear
import ttnn


class TtSwinOutput(nn.Module):
    def __init__(self, config, dim, state_dict, base_address, device):
        super().__init__()
        self.device = device

        self.dense_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.dense.weight"], self.device)
        self.dense_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.dense.bias"], self.device)

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        hidden_states = TtLinear(hidden_states, self.dense_weight, self.dense_bias)
        return hidden_states

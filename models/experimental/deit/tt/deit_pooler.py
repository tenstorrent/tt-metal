# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from torch import nn


import ttnn
from models.utility_functions import torch_to_tt_tensor_rm
from models.helper_funcs import Linear as TtLinear
from models.experimental.deit.tt.deit_config import DeiTConfig


class TtDeiTPooler(nn.Module):
    def __init__(self, config: DeiTConfig(), device, state_dict=None, base_address=""):
        super().__init__()
        dense_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.dense.weight"], device)
        dense_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.dense.bias"], device)
        self.dense = TtLinear(config.hidden_size, config.hidden_size, dense_weight, dense_bias)

        self.activation = ttnn.tanh()

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = ttnn.tanh(pooled_output)
        return pooled_output

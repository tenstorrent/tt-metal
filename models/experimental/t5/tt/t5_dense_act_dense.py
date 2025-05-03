# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from torch import nn
import ttnn
from models.utility_functions import torch2tt_tensor


class TtT5DenseActDense(nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()

        d_model = config["d_model"]
        d_ff = config["d_ff"]
        dropout_rate = config["dropout_rate"]
        self.mem_config = ttnn.L1_MEMORY_CONFIG

        # dense_act_fn = config["dense_act_fn"]

        self.out_proj_wi = torch2tt_tensor(state_dict[f"{base_address}.wi.weight"], device)
        self.out_proj_w0 = torch2tt_tensor(state_dict[f"{base_address}.wo.weight"], device)

        self.out_proj_wi = ttnn.transpose(self.out_proj_wi, -2, -1)
        self.out_proj_w0 = ttnn.transpose(self.out_proj_w0, -2, -1)

        # self.dropout = nn.Dropout(dropout_rate)
        # activation function
        self.act = ttnn.relu

    def forward(self, hidden_states):
        hidden_states = ttnn.matmul(hidden_states, self.out_proj_wi)
        hidden_states = self.act(hidden_states, output_mem_config=self.mem_config)
        # hidden_states = self.dropout(hidden_states)
        hidden_states = ttnn.matmul(hidden_states, self.out_proj_w0, memory_config=self.mem_config)
        return hidden_states

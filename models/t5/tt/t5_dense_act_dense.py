# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from torch import nn
import tt_lib
from models.utility_functions import torch2tt_tensor


# class T5DenseActDense(nn.Module):
#     # def __init__(self, config: T5Config):
#     #     super().__init__()
#     #     self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
#     #     self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
#     #     self.dropout = nn.Dropout(config.dropout_rate)
#     #     self.act = ACT2FN[config.dense_act_fn]

#     def __init__(self, d_model, d_ff, dropout_rate, dense_act_fn):
#         super().__init__()
#         self.wi = nn.Linear(d_model, d_ff, bias=False)
#         self.wo = nn.Linear(d_ff, d_model, bias=False)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.act = ACT2FN[dense_act_fn]

#     def forward(self, hidden_states):
#         hidden_states = self.wi(hidden_states)
#         hidden_states = self.act(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         if (
#             isinstance(self.wo.weight, torch.Tensor)
#             and hidden_states.dtype != self.wo.weight.dtype
#             and self.wo.weight.dtype != torch.int8
#         ):
#             hidden_states = hidden_states.to(self.wo.weight.dtype)
#         hidden_states = self.wo(hidden_states)
#         return hidden_states


class TtT5DenseActDense(nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()

        d_model = config["d_model"]
        d_ff = config["d_ff"]
        dropout_rate = config["dropout_rate"]
        self.mem_config = tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferStorage.L1)

        # dense_act_fn = config["dense_act_fn"]

        self.out_proj_wi = torch2tt_tensor(
            state_dict[f"{base_address}.wi.weight"], device
        )
        self.out_proj_w0 = torch2tt_tensor(
            state_dict[f"{base_address}.wo.weight"], device
        )

        self.out_proj_wi = tt_lib.tensor.transpose(self.out_proj_wi)
        self.out_proj_w0 = tt_lib.tensor.transpose(self.out_proj_w0)

        # self.dropout = nn.Dropout(dropout_rate)
        # activation function
        self.act = tt_lib.tensor.relu

    def forward(self, hidden_states):
        hidden_states = tt_lib.tensor.matmul(hidden_states, self.out_proj_wi)
        hidden_states = self.act(hidden_states, output_mem_config = self.mem_config)
        # hidden_states = self.dropout(hidden_states)
        hidden_states = tt_lib.tensor.matmul(hidden_states, self.out_proj_w0, self.mem_config)
        return hidden_states

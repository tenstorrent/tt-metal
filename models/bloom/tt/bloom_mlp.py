"""
SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

SPDX-License-Identifier: Apache-2.0
"""

import torch.nn as nn

import tt_lib
from models.utility_functions import torch_to_tt_tensor_rm
from models.helper_funcs import Linear as TtLinear


class TtBloomMLP(nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()

        self.config = config
        self.state_dict = state_dict
        self.base_address = base_address
        self.device = device

        self.hidden_size = config.hidden_size
        self.training = False

        self.dense_h_to_4h_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.dense_h_to_4h.weight"], self.device
        )

        self.dense_h_to_4h_bias = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.dense_h_to_4h.bias"], self.device
        )

        self.dense_h_to_4h = TtLinear(
            self.dense_h_to_4h_weight.shape()[-1],
            self.dense_h_to_4h_weight.shape()[-2],
            self.dense_h_to_4h_weight,
            self.dense_h_to_4h_bias,
        )

        self.dense_4h_to_h_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.dense_4h_to_h.weight"], self.device
        )

        self.dense_4h_to_h_bias = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.dense_4h_to_h.bias"], self.device
        )

        self.dense_4h_to_h = TtLinear(
            self.dense_4h_to_h_weight.shape()[-1],
            self.dense_4h_to_h_weight.shape()[-2],
            self.dense_4h_to_h_weight,
            self.dense_4h_to_h_bias,
        )

    def forward(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        residual: tt_lib.tensor.Tensor,
    ) -> tt_lib.tensor.Tensor:
        hidden_states = tt_lib.tensor.gelu(self.dense_h_to_4h(hidden_states))
        intermediate_output = self.dense_4h_to_h(hidden_states)
        output = tt_lib.tensor.add(intermediate_output, residual)

        return output

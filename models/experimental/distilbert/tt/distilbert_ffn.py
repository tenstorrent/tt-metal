# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

from models.utility_functions import (
    torch_to_tt_tensor_rm,
)
import ttnn
from models.helper_funcs import Linear as TtLinear


class TtFFN(nn.Module):
    def __init__(self, config, state_dict=None, base_address="", device=None):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = self.config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.device = device

        self.linear_1_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.ffn.lin1.weight"], self.device)
        self.linear_1_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.ffn.lin1.bias"], self.device)
        self.linear1 = TtLinear(
            self.linear_1_weight.get_legacy_shape()[-1],
            self.linear_1_weight.get_legacy_shape()[-2],
            self.linear_1_weight,
            self.linear_1_bias,
        )

        self.linear_2_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.ffn.lin2.weight"], self.device)
        self.linear_2_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.ffn.lin2.bias"], self.device)
        self.linear2 = TtLinear(
            self.linear_2_weight.get_legacy_shape()[-1],
            self.linear_2_weight.get_legacy_shape()[-2],
            self.linear_2_weight,
            self.linear_2_bias,
        )

        self.activation = ttnn.gelu

    def forward(self, input: ttnn.Tensor) -> ttnn.Tensor:
        x = self.linear1(input)
        x = self.activation(x)
        x = self.linear2(x)

        return x

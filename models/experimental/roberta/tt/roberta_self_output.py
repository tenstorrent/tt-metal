# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

from functools import partial

import ttnn
import tt_lib

from models.helper_funcs import Linear as TTLinear
from models.utility_functions import (
    pad_by_zero,
)


class TtRobertaSelfOutput(nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()
        self.mem_config = tt_lib.tensor.MemoryConfig(
            tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1
        )
        self.device = device

        self.dense_weight = pad_by_zero(state_dict[f"{base_address}.dense.weight"], self.device)[0]
        self.dense_bias = pad_by_zero(state_dict[f"{base_address}.dense.bias"], self.device)[0]

        gamma = pad_by_zero(state_dict[f"{base_address}.LayerNorm.weight"], self.device)[0]
        beta = pad_by_zero(state_dict[f"{base_address}.LayerNorm.bias"], self.device)[0]
        self.LayerNorm = partial(ttnn.layer_norm, epsilon=config.layer_norm_eps, weight=gamma, bias=beta)

        # TODO: Add dropout when supported
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense_linear = TTLinear(
            self.dense_weight.get_legacy_shape()[-1],
            self.dense_weight.get_legacy_shape()[-2],
            self.dense_weight,
            self.dense_bias,
        )

    def linear(self, x, weight, bias):
        weight = ttnn.transpose(weight, -2, -1)
        x = ttnn.matmul(x, weight, memory_config=self.mem_config)
        x = ttnn.add(
            x,
            bias,
            memory_config=self.mem_config,
        )
        return x

    def forward(self, hidden_states: tt_lib.tensor.Tensor, input_tensor: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        hidden_states = self.dense_linear(hidden_states)
        # TODO: Add dropout when supported
        # hidden_states = self.dropout(hidden_states)
        hidden_states = ttnn.add(hidden_states, input_tensor, memory_config=self.mem_config)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

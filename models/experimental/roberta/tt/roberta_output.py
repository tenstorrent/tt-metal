# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn

from functools import partial

import ttnn

from models.helper_funcs import Linear as TTLinear
from models.utility_functions import (
    pad_by_zero,
)


# Copied from transformers.models.bert.modeling_bert.BertOutput
class TtRobertaOutput(nn.Module):
    def __init__(
        self,
        config,
        state_dict,
        base_address,
        device,
    ):
        super().__init__()
        self.mem_config = ttnn.L1_MEMORY_CONFIG
        self.device = device

        self.dense_weight = pad_by_zero(state_dict[f"{base_address}.dense.weight"], self.device)[0]
        self.dense_bias = pad_by_zero(state_dict[f"{base_address}.dense.bias"], self.device)[0]

        gamma = pad_by_zero(state_dict[f"{base_address}.LayerNorm.weight"], self.device)[0]
        beta = pad_by_zero(state_dict[f"{base_address}.LayerNorm.bias"], self.device)[0]

        self.LayerNorm = self.LayerNorm = partial(
            ttnn.layer_norm, epsilon=config.layer_norm_eps, weight=gamma, bias=beta
        )

        # TODO: Add dropout when supported
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense_linear = TTLinear(
            self.dense_weight.shape.with_tile_padding()[-1],
            self.dense_weight.shape.with_tile_padding()[-2],
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

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_linear(hidden_states)
        # TODO: Add dropout when supported
        # hidden_states = self.dropout(hidden_states)
        hidden_states = ttnn.add(hidden_states, input_tensor, memory_config=self.mem_config)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

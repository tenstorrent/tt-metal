# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn

from functools import partial

import tt_lib

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
        self.mem_config = tt_lib.tensor.MemoryConfig(
            tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1
        )
        self.device = device

        self.dense_weight = pad_by_zero(state_dict[f"{base_address}.dense.weight"], self.device)[0]
        self.dense_bias = pad_by_zero(state_dict[f"{base_address}.dense.bias"], self.device)[0]

        gamma = pad_by_zero(state_dict[f"{base_address}.LayerNorm.weight"], self.device)[0]
        beta = pad_by_zero(state_dict[f"{base_address}.LayerNorm.bias"], self.device)[0]

        self.LayerNorm = self.LayerNorm = partial(
            tt_lib.tensor.layernorm, eps=config.layer_norm_eps, gamma=gamma, beta=beta
        )

        # TODO: Add dropout when supported
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense_linear = TTLinear(
            self.dense_weight.get_legacy_shape()[-1],
            self.dense_weight.get_legacy_shape()[-2],
            self.dense_weight,
            self.dense_bias,
        )

    def linear(self, x, weight, bias):
        weight = tt_lib.tensor.transpose(weight, -2, -1)
        x = tt_lib.tensor.matmul(x, weight, output_mem_config=self.mem_config)
        x = tt_lib.tensor.bcast(
            x,
            bias,
            tt_lib.tensor.BcastOpMath.ADD,
            tt_lib.tensor.BcastOpDim.H,
            output_mem_config=self.mem_config,
        )
        return x

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_linear(hidden_states)
        # TODO: Add dropout when supported
        # hidden_states = self.dropout(hidden_states)
        hidden_states = tt_lib.tensor.add(hidden_states, input_tensor, output_mem_config=self.mem_config)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

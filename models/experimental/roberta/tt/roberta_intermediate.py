# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

import ttnn

from models.helper_funcs import Linear as TTLinear
from models.utility_functions import (
    tt2torch_tensor,
    pad_by_zero,
)
from models.experimental.roberta.roberta_common import torch2tt_tensor


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class TtRobertaIntermediate(nn.Module):
    def __init__(self, config, state_dict, base_address, device, fall_back_to_torch_gelu=True):
        super().__init__()
        self.mem_config = ttnn.L1_MEMORY_CONFIG
        self.device = device

        self.fall_back_to_torch_gelu = fall_back_to_torch_gelu

        self.dense_weight = pad_by_zero(state_dict[f"{base_address}.dense.weight"], self.device)[0]
        self.dense_bias = pad_by_zero(state_dict[f"{base_address}.dense.bias"], self.device)[0]
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_linear(hidden_states)
        if self.fall_back_to_torch_gelu:
            torch_hidden_states = tt2torch_tensor(hidden_states)
            torch_hidden_states = torch.nn.functional.gelu(torch_hidden_states)
            hidden_states = torch2tt_tensor(torch_hidden_states, self.device)
        else:
            hidden_states = ttnn.gelu(hidden_states, memory_config=self.mem_config)
        return hidden_states

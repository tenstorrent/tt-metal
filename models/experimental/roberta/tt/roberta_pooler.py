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


# Copied from transformers.models.bert.modeling_bert.BertPooler
class TtRobertaPooler(nn.Module):
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
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        # Convert to torch to slice the tensor
        hidden_states_to_torch = tt2torch_tensor(hidden_states)
        hidden_states_to_torch = hidden_states_to_torch.squeeze(0)
        first_token_tensor = hidden_states_to_torch[:, 0]
        tt_first_token_tensor = torch2tt_tensor(first_token_tensor, self.device)

        pooled_output = self.dense_linear(tt_first_token_tensor)
        pooled_output = ttnn.tanh(pooled_output, memory_config=self.mem_config)
        return pooled_output

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from loguru import logger
import math

from transformers import T5Model
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
)


def gelu_new(x, device):
    x = tt2torch_tensor(x)
    x = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    x = torch2tt_tensor(x, device)

    return x


class TtT5DenseGatedActDense(torch.nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()

        self.device = device
        self.mem_config = ttnn.L1_MEMORY_CONFIG

        enc_dec = "decoder" if config["is_decoder"] else "encoder"

        self.wi_0_weights = torch2tt_tensor(state_dict[f"{base_address}.wi_0.weight"], device)
        self.wi_1_weights = torch2tt_tensor(state_dict[f"{base_address}.wi_1.weight"], device)
        self.wo_weights = torch2tt_tensor(state_dict[f"{base_address}.wo.weight"], device)

        self.wi_0_weights = ttnn.transpose(self.wi_0_weights, -2, -1)
        self.wi_1_weights = ttnn.transpose(self.wi_1_weights, -2, -1)
        self.wo_weights = ttnn.transpose(self.wo_weights, -2, -1)

        # self.dropout = nn.Dropout(config["dropout_rate"])
        self.act = gelu_new

    def forward(self, hidden_states):
        hidden_gelu = self.act(ttnn.matmul(hidden_states, self.wi_0_weights), self.device)
        hidden_linear = ttnn.matmul(hidden_states, self.wi_1_weights, memory_config=self.mem_config)
        hidden_states = ttnn.mul(hidden_gelu, hidden_linear, memory_config=self.mem_config)
        # hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        # if (
        #    isinstance(self.wo.weight, torch.Tensor)
        #    and hidden_states.dtype != self.wo.weight.dtype
        #    and self.wo.weight.dtype != torch.int8
        # ):
        #    hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = ttnn.matmul(hidden_states, self.wo_weights, memory_config=self.mem_config)
        return hidden_states

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import ttnn

from models.experimental.llama.llama_utils import linear
from models.utility_functions import torch_to_tt_tensor_rm


class TtLlamaMLP(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.bias = None

        self.out_gate_proj = torch_to_tt_tensor_rm(
            self.state_dict[f"{base_url}.{layer_num}.mlp.gate_proj.weight"], self.device
        )
        self.out_down_proj = torch_to_tt_tensor_rm(
            self.state_dict[f"{base_url}.{layer_num}.mlp.down_proj.weight"], self.device
        )
        self.out_up_proj = torch_to_tt_tensor_rm(
            self.state_dict[f"{base_url}.{layer_num}.mlp.up_proj.weight"], self.device
        )

        if hidden_act == "silu":  # silu
            self.act_fn = ttnn.silu

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # gate proj
        gate = linear(x, self.out_gate_proj, self.bias, self.device)
        # apply silu activation function
        gate = self.act_fn(gate)

        # up proj
        up = linear(x, self.out_up_proj, self.bias, self.device)

        # product
        prod = ttnn.mul(gate, up)

        # down
        hidden_states = linear(prod, self.out_down_proj, self.bias, self.device)

        # return TT Tensor
        return hidden_states

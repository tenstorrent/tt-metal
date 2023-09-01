# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import tt_lib
import models.nanogpt.tt.nanogpt_mlp as nanogpt_mlp
import models.nanogpt.tt.nanogpt_attention as nanogpt_attention


from models.utility_functions import (
    torch_to_tt_tensor_rm,
)


class TtBlock(nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()

        self.device = device
        self.config = config

        self.beta_1 = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.ln_1.bias"], self.device
        )

        self.gamma_1 = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.ln_1.weight"], self.device
        )

        self.ln_1 = tt_lib.tensor.layernorm

        self.attn = nanogpt_attention.TtCausalSelfAttention(
            config, state_dict, f"{base_address}.attn", device
        )

        self.beta_2 = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.ln_2.bias"], self.device
        )

        self.gamma_2 = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.ln_2.weight"], self.device
        )

        self.ln_2 = tt_lib.tensor.layernorm

        self.mlp = nanogpt_mlp.TtMLP(
            f"{base_address}.mlp", self.config, state_dict, device
        )

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        tmp = self.attn.forward(
            self.ln_1(x, eps=1e-5, gamma=self.gamma_1, beta=self.beta_1)
        )
        x = tt_lib.tensor.add(x, tmp)

        tmp = self.mlp.forward(
            self.ln_2(x, eps=1e-5, gamma=self.gamma_2, beta=self.beta_2)
        )
        x = tt_lib.tensor.add(x, tmp)

        return x

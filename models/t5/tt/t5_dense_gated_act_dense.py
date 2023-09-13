# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib

from models.utility_functions import (
    torch_to_tt_tensor_rm,
)
from models.helper_funcs import Linear as TtLinear


class TtT5DenseGatedActDense(torch.nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()

        self.device = device

        self.out_mem_config_l1 = tt_lib.tensor.MemoryConfig(
            True, tt_lib.tensor.BufferType.L1
        )

        self.wi_0_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.wi_0.weight"], device, put_on_device=True
        )
        self.wi_0 = TtLinear(
            config.d_model,
            config.d_ff,
            self.wi_0_weights,
            output_mem_config=self.out_mem_config_l1,
        )

        self.wi_1_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.wi_1.weight"], device, put_on_device=True
        )
        self.wi_1 = TtLinear(
            config.d_model,
            config.d_ff,
            self.wi_1_weights,
            output_mem_config=self.out_mem_config_l1,
        )

        self.wo_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.wo.weight"], device, put_on_device=True
        )
        self.wo = TtLinear(
            config.d_ff,
            config.d_model,
            self.wo_weights,
            output_mem_config=self.out_mem_config_l1,
        )

        self.act = tt_lib.tensor.gelu

    def forward(self, hidden_states: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = tt_lib.tensor.mul(
            hidden_gelu, hidden_linear, output_mem_config=self.out_mem_config_l1
        )
        hidden_states = self.wo(hidden_states)
        return hidden_states

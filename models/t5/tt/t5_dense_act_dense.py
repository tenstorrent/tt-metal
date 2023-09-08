# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from torch import nn
import tt_lib
from models.utility_functions import torch_to_tt_tensor_rm
from models.helper_funcs import Linear as TtLinear


class TtT5DenseActDense(nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()

        self.out_proj_wi_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.wi.weight"], device, put_on_device=True
        )
        self.out_proj_wi = TtLinear(
            config.d_model,
            config.d_ff,
            self.out_proj_wi_weights,
        )

        self.out_proj_w0_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.wo.weight"], device, put_on_device=True
        )
        self.out_proj_w0 = TtLinear(
            config.d_ff,
            config.d_model,
            self.out_proj_w0_weights,
        )

        self.act = tt_lib.tensor.relu

    def forward(self, hidden_states: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        hidden_states = self.out_proj_wi(hidden_states)
        hidden_states = self.act(
            hidden_states,
        )
        hidden_states = self.out_proj_w0(hidden_states)
        return hidden_states

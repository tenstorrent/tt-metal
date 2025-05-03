# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from torch import nn

import ttnn

from models.experimental.deit.tt.activations import ACT2FN
from models.experimental.deit.tt.deit_config import DeiTConfig
from models.utility_functions import torch_to_tt_tensor_rm
from models.helper_funcs import Linear as TtLinear


class TtDeiTIntermediate(nn.Module):
    def __init__(self, config: DeiTConfig(), device, state_dict=None, base_address="") -> None:
        super().__init__()

        dense_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.dense.weight"], device)
        dense_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.dense.bias"], device)
        self.dense = TtLinear(config.hidden_size, config.intermediate_size, dense_weight, dense_bias)

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: ttnn.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

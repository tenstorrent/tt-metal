# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from torch import nn
from models.common.utility_functions import torch2tt_tensor

import ttnn


class TtLlamaRMSNorm(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        layer_num,
        layer_position,
        hidden_size,
        eps=1e-6,
    ):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()

        self.device = device
        self.variance_epsilon = eps
        self.state_dict = state_dict

        # check if it is final norm layer
        if layer_num is not None:
            pytorch_weights = self.state_dict[f"{base_url}.{layer_num}.{layer_position}.weight"]
        else:
            pytorch_weights = self.state_dict[f"model.norm.weight"]

        # get weights
        self.weight = torch2tt_tensor(pytorch_weights, self.device)

    def forward(self, hidden_states):
        return ttnn.rms_norm(hidden_states, epsilon=self.variance_epsilon, weight=self.weight)

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
from models.utility_functions import pad_by_zero

# class T5LayerNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-6):
#         """
#         Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
#         """
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, hidden_states):
#         # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
#         # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
#         # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
#         # half-precision inputs is done in fp32

#         variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)

#         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#         t = torch.rsqrt(variance + self.variance_epsilon)

#         # convert into half-precision if necessary
#         if self.weight.dtype in [torch.float16, torch.bfloat16]:
#             hidden_states = hidden_states.to(self.weight.dtype)
#         return self.weight * hidden_states


class TtT5LayerNorm(torch.nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.variance_epsilon = config["layer_norm_epsilon"]
        self.device = device

        # get weights
        pytorch_weights = state_dict[f"{base_address}.weight"]

        self.weight = pad_by_zero(pytorch_weights, device)[0]

    def forward(self, hidden_states):
        return tt_lib.tensor.rmsnorm(hidden_states, self.variance_epsilon, self.weight)

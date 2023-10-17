# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import tt_lib
from models.mistral.tt.mistral_configuration import TtModelArgs
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.helper_funcs import Linear as TtLinear


class TtFeedForward(nn.Module):
    def __init__(
        self,
        args: TtModelArgs,
        base_address=None,
        device=None,
        state_dict=None,
    ):
        super().__init__()
        self.device = device
        self.w1_weights = torch_to_tt_tensor_rm(state_dict[f"{base_address}w1.weight"], self.device)
        self.w1 = TtLinear(
            args.dim,
            args.hidden_dim,
            self.w1_weights,
        )

        self.w2_weights = torch_to_tt_tensor_rm(state_dict[f"{base_address}w2.weight"], self.device)
        self.w2 = TtLinear(
            args.hidden_dim,
            args.dim,
            self.w2_weights,
        )

        self.w3_weights = torch_to_tt_tensor_rm(state_dict[f"{base_address}w3.weight"], self.device)
        self.w3 = TtLinear(args.dim, args.hidden_dim, self.w3_weights)

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        x = tt_lib.tensor.mul(self.w1(x), self.w3(x))
        x = tt_lib.tensor.silu(x)
        return self.w2(x)

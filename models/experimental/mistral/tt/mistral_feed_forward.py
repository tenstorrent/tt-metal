# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import tt_lib
from models.experimental.mistral.tt.mistral_configuration import TtModelArgs
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.experimental.mistral.mistral_helper_funcs import Linear as TtLinear


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
        self.w1_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}w1.weight"], self.device, put_on_device=False
        )
        self.w1 = TtLinear(
            args.dim,
            args.hidden_dim,
            self.w1_weights,
            device=self.device,
        )
        self.w2_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}w2.weight"], self.device, put_on_device=False
        )
        self.w2 = TtLinear(
            args.hidden_dim,
            args.dim,
            self.w2_weights,
            device=self.device,
        )
        self.w3_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}w3.weight"], self.device, put_on_device=False
        )
        self.w3 = TtLinear(
            args.dim,
            args.hidden_dim,
            self.w3_weights,
            device=self.device,
        )

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        silu_out = tt_lib.tensor.silu(self.w1(x))
        x = tt_lib.tensor.mul(silu_out, self.w3(x))
        return self.w2(x)

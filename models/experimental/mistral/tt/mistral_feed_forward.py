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

        w1_weights = state_dict[f"{base_address}w1.weight"]
        ref_w1_weights = w1_weights.unsqueeze(0).unsqueeze(0)
        self.w1_weights = tt_lib.tensor.Tensor(
            ref_w1_weights.reshape(-1).tolist(),
            ref_w1_weights.shape,
            args.WEIGHTS_DTYPE,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )
        self.w1 = TtLinear(
            args.dim,
            args.hidden_dim,
            self.w1_weights,
            device=self.device,
        )

        w2_weights = state_dict[f"{base_address}w2.weight"]
        ref_w2_weights = w2_weights.unsqueeze(0).unsqueeze(0)
        self.w2_weights = tt_lib.tensor.Tensor(
            ref_w2_weights.reshape(-1).tolist(),
            ref_w2_weights.shape,
            args.WEIGHTS_DTYPE,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )
        self.w2 = TtLinear(
            args.hidden_dim,
            args.dim,
            self.w2_weights,
            device=self.device,
        )

        w3_weights = state_dict[f"{base_address}w3.weight"]
        ref_w3_weights = w3_weights.unsqueeze(0).unsqueeze(0)
        self.w3_weights = tt_lib.tensor.Tensor(
            ref_w3_weights.reshape(-1).tolist(),
            ref_w3_weights.shape,
            args.WEIGHTS_DTYPE,
            tt_lib.tensor.Layout.ROW_MAJOR,
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

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
        tt_cache_path=None,
    ):
        super().__init__()
        self.device = device
        self.args = args

        self.w1_weights = tt_lib.tensor.load_tensor(
            tt_cache_path + base_address + "w1.weight" + str(self.args.WEIGHTS_DTYPE) + ".bin"
        )
        self.w1 = TtLinear(
            args.dim,
            args.hidden_dim,
            self.w1_weights,
            device=self.device,
        )

        self.w2_weights = tt_lib.tensor.load_tensor(
            tt_cache_path + base_address + "w2.weight" + str(self.args.WEIGHTS_DTYPE) + ".bin"
        )
        self.w2 = TtLinear(
            args.hidden_dim,
            args.dim,
            self.w2_weights,
            device=self.device,
        )

        self.w3_weights = tt_lib.tensor.load_tensor(
            tt_cache_path + base_address + "w3.weight" + str(self.args.WEIGHTS_DTYPE) + ".bin"
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

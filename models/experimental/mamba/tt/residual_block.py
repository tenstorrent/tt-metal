# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import tt_lib

from typing import Callable

from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.helper_funcs import Linear
from models.experimental.mamba.reference.args import ModelArgs
from models.experimental.mamba.tt.mamba_block import TtMambaBlock


class TtResidualBlock(torch.nn.Module):
    def __init__(self, args: ModelArgs, device: tt_lib.device, load_fn: Callable):
        super().__init__()

        self.device = device
        self.args = args

        rms_norm_weight_name = "norm.weight"
        self.rms_norm_weights = load_fn(rms_norm_weight_name)

        self.tt_mamba_block = TtMambaBlock(self.args, self.device, load_fn)

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        mamba_input = tt_lib.tensor.rmsnorm(x, self.args.eps, self.rms_norm_weights)
        mamba_input = self.tt_mamba_block(mamba_input)
        x = tt_lib.tensor.add(x, mamba_input)
        return x

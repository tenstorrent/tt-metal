# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

from typing import Callable

from models.experimental.mamba.reference.args import ModelArgs
from models.experimental.mamba.tt_opt.mamba_block import TtMambaBlock

class TtResidualBlock(torch.nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        device,
        configs,
        load_fn: Callable
    ):
        super().__init__()

        self.device = device
        self.args = args

        rms_norm_weight_name = "norm.weight"
        self.rms_norm_weights = load_fn(rms_norm_weight_name)

        self.tt_mamba_block = TtMambaBlock(self.args, self.device, configs, load_fn)

    def forward(self, x):
        mamba_input = x
        rms_norm_weights = ttnn.to_memory_config(self.rms_norm_weights, memory_config=ttnn.L1_MEMORY_CONFIG)
        mamba_input = ttnn.rms_norm(x, rms_norm_weights, epsilon=self.args.eps)
        ttnn.deallocate(rms_norm_weights)
        mamba_input = self.tt_mamba_block(mamba_input)
        x = ttnn.add(x, mamba_input)
        return x

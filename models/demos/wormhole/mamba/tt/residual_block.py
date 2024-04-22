# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

from typing import Callable

from models.demos.wormhole.mamba.reference.args import ModelArgs
from models.demos.wormhole.mamba.tt.mamba_block import TtMambaBlock
from models.demos.wormhole.mamba.tt.transforms import MambaSsmBlockTransformer


class TtResidualBlock(torch.nn.Module):
    def __init__(self, args: ModelArgs, device, configs, load_fn: Callable, transformer: MambaSsmBlockTransformer):
        super().__init__()

        self.device = device
        self.args = args

        rms_norm_weight_name = "norm.weight"
        self.rms_norm_weights = load_fn(rms_norm_weight_name)

        self.tt_mamba_block = TtMambaBlock(self.args, self.device, configs, load_fn, transformer)

    def forward(self, x):
        assert len(x.shape) == 4, "Mamba residual block expects inputs to be rank 4"

        residual = x
        rms_norm_weights = ttnn.to_memory_config(self.rms_norm_weights, memory_config=ttnn.L1_MEMORY_CONFIG)
        mamba_x = ttnn.rms_norm(x, rms_norm_weights, epsilon=self.args.eps)
        ttnn.deallocate(rms_norm_weights)

        mamba_x = self.tt_mamba_block(mamba_x)

        return ttnn.add(residual, mamba_x)

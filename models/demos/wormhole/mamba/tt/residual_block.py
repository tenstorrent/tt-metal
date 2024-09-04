# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

from typing import Callable

from models.demos.wormhole.mamba.reference.args import ModelArgs
from models.demos.wormhole.mamba.tt.mamba_block import TtMambaBlock


class TtResidualBlock(torch.nn.Module):
    def __init__(self, args: ModelArgs, device, configs, load_fn: Callable):
        super().__init__()

        self.device = device
        self.args = args
        self.configs = configs

        rms_norm_weight_name = "norm.weight"
        self.rms_norm_weights = load_fn(rms_norm_weight_name)

        self.tt_mamba_block = TtMambaBlock(self.args, self.device, configs, load_fn)

    def to_prefill(self, prefill_config):
        self.configs = prefill_config
        self.tt_mamba_block.to_prefill(prefill_config)

    def to_decode(self, decode_config):
        self.configs = decode_config
        self.tt_mamba_block.to_decode(decode_config)

    def forward(self, x):
        assert len(x.shape) == 4, "Mamba residual block expects inputs to be rank 4"

        residual = x
        residual = ttnn.to_memory_config(residual, ttnn.DRAM_MEMORY_CONFIG)

        rms_norm_weights = ttnn.to_memory_config(self.rms_norm_weights, memory_config=ttnn.L1_MEMORY_CONFIG)
        x_sharded = ttnn.interleaved_to_sharded(x, self.configs["sharded_h"])
        ttnn.deallocate(x)
        mamba_x = ttnn.rms_norm(
            x_sharded,
            epsilon=self.args.eps,
            weight=rms_norm_weights,
            program_config=self.configs["SHARDED_NORM_PRGM_CFG"],
            memory_config=self.configs["sharded_h"],
        )
        ttnn.deallocate(x_sharded)
        mamba_x_in_l1_int = ttnn.to_memory_config(mamba_x, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(mamba_x)
        ttnn.deallocate(rms_norm_weights)

        mamba_block_out = self.tt_mamba_block(mamba_x_in_l1_int)
        ttnn.deallocate(mamba_x_in_l1_int)

        return ttnn.add(
            residual,
            mamba_block_out,
            dtype=self.configs["dtype"]["activations"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tensor=mamba_block_out,
        )

    def reset(self):
        self.tt_mamba_block.reset()

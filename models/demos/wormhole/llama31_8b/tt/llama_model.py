# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import os
import ttnn
import torch
import torch.nn as nn
from models.demos.wormhole.llama31_8b.tt.llama_decoder import TtTransformerBlock
from models.common.rmsnorm import RMSNorm
import ttnn
from typing import Optional


class TtTransformer(nn.Module):
    def __init__(
        self,
        args,
        dtype,
        device,
        state_dict,
        weight_cache_path,
        layers,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.device = device
        self.dtype = dtype
        self.model_config = args.get_model_config()
        self.grid_size = self.args.max_grid_size
        assert self.vocab_size > 0

        self.layers = torch.nn.ModuleList(
            [
                TtTransformerBlock(
                    args=args,
                    device=device,
                    dtype=dtype,
                    state_dict=state_dict,
                    weight_cache_path=weight_cache_path,
                    layer_num=i,
                )
                for i in layers
            ]
        )
        self.norm = RMSNorm(
            device=device,
            dim=args.dim,
            state_dict=state_dict,
            layer_num=None,
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=dtype,
            weight_key="norm",
        )

        # Helper function to create output memory config
        def create_output_mem_config(size):
            padded_size = math.ceil(size / (32 * 12)) * (32 * 12)
            shard_spec = ttnn.ShardSpec(
                args.dram_weight_grid, (4096, padded_size // 12), ttnn.ShardOrientation.ROW_MAJOR, False
            )
            return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

        # Split the output weight
        split_size = self.vocab_size // 2
        output_weight = state_dict["output.weight"].permute(1, 0)
        output_weight_1 = output_weight[:, :split_size]
        output_weight_2 = output_weight[:, split_size:]

        # Create ttnn tensors for split weights
        self.output_weight_1 = ttnn.as_tensor(
            output_weight_1,
            device=device,
            memory_config=create_output_mem_config(split_size),
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            cache_file_name=None if args.dummy_weights else weight_cache_path / "output_sharded_1",
        )
        self.output_weight_2 = ttnn.as_tensor(
            output_weight_2,
            device=device,
            memory_config=create_output_mem_config(split_size),
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            cache_file_name=None if args.dummy_weights else weight_cache_path / "output_sharded_2",
        )

        self.compute_kernel_config_output = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,  # Save as much L1 as possible
            packer_l1_acc=True,
        )

        self.program_config_output = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=1,
            per_core_M=1,
            per_core_N=32,  # vocab_size / 2 / tile_size / core_count
            fused_activation=None,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mat=None,
        transformation_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        get_last_token=-1,
    ):
        for layer in self.layers:
            x = layer(x, current_pos, rot_mat, transformation_mats, user_id, mode, page_table)
        if mode == "prefill" and get_last_token == -1:
            return x

        # Slicing the tensor to the nearest celing/floor multiples of 32 for the prefill_len, to get the last token
        if get_last_token != -1:
            x = ttnn.slice(x, (0, 0, get_last_token, 0), (1, 1, get_last_token + 32, 4096))

        x = self.norm(x)

        x = ttnn.interleaved_to_sharded(x, self.model_config["SHARDED_MLP_DECODE_INPUT_MEMCFG"])

        # Split linear operation
        output_1 = ttnn.linear(
            x,
            self.output_weight_1,
            compute_kernel_config=self.compute_kernel_config_output,
            program_config=self.program_config_output,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )
        output_1 = ttnn.sharded_to_interleaved(output_1, memory_config=ttnn.L1_MEMORY_CONFIG)

        output_2 = ttnn.linear(
            x,
            self.output_weight_2,
            compute_kernel_config=self.compute_kernel_config_output,
            program_config=self.program_config_output,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )

        output_2 = ttnn.sharded_to_interleaved(output_2, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Concatenate the outputs
        output = ttnn.concat([output_1, output_2], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

        return output

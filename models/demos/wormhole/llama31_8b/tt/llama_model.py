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

        self.lm_head = LMHead(
            args=args,
            device=device,
            dtype=dtype,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
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

        # Slicing the tensor to the nearest ceiling/floor multiples of 32 for the prefill_len, to get the last token
        if get_last_token != -1:
            x = ttnn.slice(x, (0, 0, get_last_token, 0), (1, 1, get_last_token + 32, 4096))

        x = self.norm(x)
        output = self.lm_head(x)

        return output


class LMHead(nn.Module):
    def __init__(
        self,
        args,
        device,
        dtype,
        state_dict,
        weight_cache_path,
    ):
        super().__init__()
        self.args = args
        self.device = device
        self.dtype = dtype
        self.vocab_size = args.vocab_size
        self.num_splits = 4  # 2 splits hangs after a few iterations, 1 split doesn't fit into L1

        split_size = self.vocab_size // self.num_splits

        # Helper function to create output memory config
        def create_output_mem_config(size):
            padded_size = math.ceil(size / (32 * 12)) * (32 * 12)
            shard_spec = ttnn.ShardSpec(
                args.dram_weight_grid, (4096, padded_size // 12), ttnn.ShardOrientation.ROW_MAJOR, False
            )
            return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

        # Split the output weight
        output_weight = state_dict["output.weight"].permute(1, 0)
        self.output_weights = []
        for i in range(self.num_splits):
            start = i * split_size
            end = (i + 1) * split_size
            weight_part = output_weight[:, start:end]
            self.output_weights.append(
                ttnn.as_tensor(
                    weight_part,
                    device=device,
                    memory_config=create_output_mem_config(split_size),
                    layout=ttnn.TILE_LAYOUT,
                    dtype=dtype,
                    cache_file_name=None if args.dummy_weights else weight_cache_path / f"output_sharded_{i+1}",
                )
            )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # Calculate per_core_N based on the number of splits
        tile_size = 32
        core_count = 64  # Assuming 8x8 core grid
        per_core_N = -(-split_size // (tile_size * core_count))  # Ceiling division

        self.program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=2,
            per_core_M=1,
            per_core_N=per_core_N,
            fused_activation=None,
        )

    def forward(self, x: ttnn.Tensor):
        x = ttnn.interleaved_to_sharded(x, self.args.get_model_config()["SHARDED_MLP_DECODE_INPUT_MEMCFG"])

        outputs = []
        for weight in self.output_weights:
            output = ttnn.linear(
                x,
                weight,
                compute_kernel_config=self.compute_kernel_config,
                program_config=self.program_config,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
            )
            outputs.append(ttnn.sharded_to_interleaved(output, memory_config=ttnn.L1_MEMORY_CONFIG))

        # Concatenate the outputs
        output = ttnn.concat(outputs, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

        return output

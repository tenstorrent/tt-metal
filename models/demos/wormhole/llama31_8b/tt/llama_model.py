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

        # output_weight: 4096 x 128256: width-sharded on 12 banks
        output_shard_shape = (
            4096,
            128256 // 12,
        )
        output_shard_spec = ttnn.ShardSpec(
            args.dram_weight_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False
        )
        output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, output_shard_spec
        )
        self.output_weight = ttnn.as_tensor(
            state_dict["output.weight"].permute(1, 0),
            device=device,
            memory_config=output_mem_config,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            cache_file_name=None if args.dummy_weights else weight_cache_path / "output_sharded",
        )

        self.compute_kernel_config_output = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,  # Save as much L1 as possible
            packer_l1_acc=True,
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
        output = ttnn.linear(
            x,
            self.output_weight,
            compute_kernel_config=self.compute_kernel_config_output,
            program_config=self.model_config["OUTPUT_MM_PROGCFG"],
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=self.dtype,
        )

        ttnn.deallocate(x)

        return output

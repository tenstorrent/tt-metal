# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import ttnn
import torch
import torch.nn as nn
from models.demos.wormhole.llama31_8b_N300.tt.llama_decoder import TtTransformerBlock
from models.common.rmsnorm import RMSNorm
import ttnn
from typing import Optional


class TtTransformer(nn.Module):
    def __init__(
        self,
        args,
        dtype,
        device_mesh,
        state_dict,
        weight_cache_path,
        layers,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.device_mesh = device_mesh
        self.dtype = dtype
        self.model_config = args.get_model_config()
        self.grid_size = self.args.max_grid_size
        assert self.vocab_size > 0

        self.layers = torch.nn.ModuleList(
            [
                TtTransformerBlock(
                    args=args,
                    device_mesh=device_mesh,
                    dtype=dtype,
                    state_dict=state_dict,
                    weight_cache_path=weight_cache_path,
                    layer_num=i,
                )
                for i in layers
            ]
        )
        self.norm = RMSNorm(
            device=device_mesh,
            dim=args.dim,
            state_dict=state_dict,
            layer_num=None,
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=dtype,
            weight_key="norm",
        )

        self.output_weight = ttnn.as_tensor(
            state_dict["output.weight"].permute(1, 0),
            device=device_mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(device_mesh, dim=-1),
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=None if args.dummy_weights else weight_cache_path / "output.weight",
        )

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        current_pos_attn,
        rot_mat=None,
        transformation_mats=None,
        user_id=0,
        mode="decode",
        get_last_token=-1,
    ):
        for layer in self.layers:
            x = layer(x, current_pos, current_pos_attn, rot_mat, transformation_mats, user_id, mode)
        if mode == "prefill " and get_last_token == -1:
            return x

        # slicing for the last token
        if get_last_token != -1:
            x = ttnn.slice(x, ttnn.Shape((0, 0, get_last_token, 0)), ttnn.Shape((0, 0, get_last_token + 31, 4095)))

        x = self.norm(x)

        output = ttnn.linear(
            x,
            self.output_weight,
            compute_kernel_config=self.args.get_compute_kernel_config(),
            program_config=self.model_config["OUTPUT_MM_PROGCFG"],
            memory_config=self.model_config["OUTPUT_MM_MEMCFG"],
            dtype=self.dtype,
        )

        return output

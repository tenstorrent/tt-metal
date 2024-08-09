# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

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
        rot_mat,
        start_pos,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.start_pos = start_pos
        self.device = device
        self.model_config = args.get_model_config()
        self.dtype = dtype
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
                    rot_mat=rot_mat,
                    start_pos=start_pos,
                )
                for i in layers
            ]
        )
        self.norm = RMSNorm(
            device=device,
            dim=args.dim,
            state_dict=state_dict,
            layer_num=None,
            weight_cache_path=weight_cache_path,
            weight_dtype=dtype,
            weight_key="norm",
        )

        # Divide the output weight in half due to very large vocab size (128k)
        self.output_weight_1 = ttnn.as_tensor(
            state_dict["output.weight"].permute(1, 0)[:, : self.vocab_size // 2],
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=weight_cache_path / "output_1half.weight",
        )
        self.output_weight_2 = ttnn.as_tensor(
            state_dict["output.weight"].permute(1, 0)[:, self.vocab_size // 2 :],
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=weight_cache_path / "output_2half.weight",
        )

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos: int,
        attn_masks: Optional[ttnn.Tensor] = None,
        rot_mat=None,
        transformation_mats=None,
        user_id=0,
        mode="decode",
    ):
        for layer in self.layers:
            x = layer(x, current_pos, attn_masks, rot_mat, transformation_mats, user_id, mode)
        if mode == "prefill":
            return x
        x = self.norm(x)

        # Divide the output weight in half, and do 2 matmuls + 1 concat, due to very large vocab size (128k) not fitting in L1 on 1-chip
        output_1 = ttnn.linear(
            x,
            self.output_weight_1,
            memory_config=self.model_config["OUTPUT_MM_MEMCFG"],
            compute_kernel_config=self.args.get_compute_kernel_config(),
            dtype=self.dtype,
            core_grid=self.args.max_grid_size,
        )
        output_2 = ttnn.linear(
            x,
            self.output_weight_2,
            memory_config=self.model_config["OUTPUT_MM_MEMCFG"],
            compute_kernel_config=self.args.get_compute_kernel_config(),
            dtype=self.dtype,
            core_grid=self.args.max_grid_size,
        )
        output = ttnn.concat([output_1, output_2], dim=-1)

        output_1.deallocate()
        output_2.deallocate()

        return output

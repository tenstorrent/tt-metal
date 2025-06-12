# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn
import re

from models.experimental.stable_diffusion_xl_base.tt.tt_transformerblock import TtBasicTransformerBlock
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import (
    prepare_gn_mask,
    prepare_gn_beta_gamma,
    prepare_linear_params,
)


class TtTransformer2DModel(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        model_config,
        query_dim,
        num_attn_heads,
        out_dim,
        weights_dtype=ttnn.bfloat16,
    ):
        super().__init__()

        self.device = device

        self.norm_core_grid = ttnn.CoreGrid(y=8, x=8)
        self.norm_groups = 32
        self.norm_eps = 1e-6

        pattern = re.compile(rf"^{re.escape(module_path)}\.transformer_blocks\.(\d+)")
        transformer_blocks = set(int(match.group(1)) for key in state_dict.keys() if (match := pattern.match(key)))
        self.num_layers = len(transformer_blocks)

        self.transformer_blocks = []
        for i in range(self.num_layers):
            self.transformer_blocks.append(
                TtBasicTransformerBlock(
                    device,
                    state_dict,
                    f"{module_path}.transformer_blocks.{i}",
                    model_config,
                    query_dim,
                    num_attn_heads,
                    out_dim,
                    weights_dtype=weights_dtype,
                )
            )

        norm_weights = state_dict[f"{module_path}.norm.weight"]
        norm_bias = state_dict[f"{module_path}.norm.bias"]
        self.gamma_t, self.beta_t = prepare_gn_beta_gamma(device, norm_weights, norm_bias, self.norm_core_grid.y)
        self.input_mask = prepare_gn_mask(self.device, norm_weights.shape[0], self.norm_groups, self.norm_core_grid.y)

        weights = state_dict[f"{module_path}.proj_in.weight"].unsqueeze(0).unsqueeze(0)
        bias = state_dict[f"{module_path}.proj_in.bias"]
        self.tt_weights_in, self.tt_bias_in = prepare_linear_params(device, weights, bias, weights_dtype)

        weights = state_dict[f"{module_path}.proj_out.weight"].unsqueeze(0).unsqueeze(0)
        bias = state_dict[f"{module_path}.proj_out.bias"]
        self.tt_weights_out, self.tt_bias_out = prepare_linear_params(device, weights, bias, weights_dtype)

        self.program_config_in = model_config.get_matmul_config(matmul_path=f"{module_path}.proj_in")
        self.compute_config_in = model_config.get_mm_compute_config(f"{module_path}.proj_in")
        self.program_config_out = model_config.get_matmul_config(matmul_path=f"{module_path}.proj_out")
        self.compute_config_out = model_config.get_mm_compute_config(f"{module_path}.proj_out")

    def forward(self, input_tensor, input_shape, attention_mask=None, encoder_hidden_states=None):
        B, C, H, W = input_shape
        hidden_states = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)

        grid_coord = ttnn.CoreCoord(self.norm_core_grid.x - 1, self.norm_core_grid.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        shard_shape = B * H * W // self.norm_core_grid.x, C // self.norm_core_grid.y
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
        sharded_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        hidden_states = ttnn.to_memory_config(hidden_states, sharded_mem_config)

        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=self.norm_groups,
            input_mask=self.input_mask,
            weight=self.gamma_t,
            bias=self.beta_t,
            memory_config=sharded_mem_config,
            core_grid=self.norm_core_grid,
            epsilon=self.norm_eps,
        )

        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)

        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_weights_in,
            bias=self.tt_bias_in,
            program_config=self.program_config_in,
            compute_kernel_config=self.compute_config_in,
        )

        for i, transformer_block in enumerate(self.transformer_blocks):
            hidden_states = transformer_block(hidden_states, attention_mask, encoder_hidden_states)

        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_weights_out,
            bias=self.tt_bias_out,
            program_config=self.program_config_out,
            compute_kernel_config=self.compute_config_out,
        )

        hidden_states = ttnn.add(hidden_states, input_tensor)

        return hidden_states

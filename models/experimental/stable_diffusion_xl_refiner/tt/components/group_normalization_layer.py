# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from dataclasses import dataclass
from typing import Tuple
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import (
    prepare_gn_mask,
    prepare_gn_beta_gamma,
)


@dataclass
class NormConfig:
    num_groups: int = 32
    eps: float = 1e-5
    num_out_blocks: int = 1
    core_grid: ttnn.CoreGrid = None
    sharded: bool = True

    def __post_init__(self):
        if self.core_grid is None:
            self.core_grid = ttnn.CoreGrid(y=8, x=8)


def make_norm_config(
    sharded: bool = True, num_out_blocks: int = 1, core_grid: Tuple[int, int] = (8, 8), eps: float = 1e-5
) -> NormConfig:
    config = NormConfig(
        sharded=sharded, num_out_blocks=num_out_blocks, core_grid=ttnn.CoreGrid(y=core_grid[0], x=core_grid[1]), eps=eps
    )
    return config


class GroupNormalizationLayer:
    def __init__(self, device, weights, bias, norm_config=make_norm_config()):
        self.device = device

        # From norm_config
        self.sharded = norm_config.sharded
        self.norm_groups = norm_config.num_groups
        self.norm_eps = norm_config.eps
        self.num_out_blocks = norm_config.num_out_blocks

        if self.sharded:
            self.core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            self.core_grid = norm_config.core_grid

        # Prepare normalization parameters
        self.gamma_t, self.beta_t = prepare_gn_beta_gamma(device, weights, bias, self.core_grid.y)
        self.input_mask = prepare_gn_mask(device, weights.shape[0], self.norm_groups, self.core_grid.y)

    def forward(self, hidden_states, B, C, H, W):
        if self.sharded:
            # Configure sharded memory layout
            grid_coord = ttnn.CoreCoord(self.core_grid.x - 1, self.core_grid.y - 1)
            shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
            shard_shape = B * H * W // self.core_grid.x, C // self.core_grid.y
            shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
            memory_config = ttnn.MemoryConfig(
                ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
            )

            # Convert to sharded layout
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
            hidden_states = ttnn.to_memory_config(hidden_states, memory_config)

            inplace = True
            num_out_blocks = None
        else:
            # Use DRAM memory config
            memory_config = ttnn.DRAM_MEMORY_CONFIG
            inplace = False
            num_out_blocks = self.num_out_blocks

        # Apply group normalization
        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=self.norm_groups,
            input_mask=self.input_mask,
            weight=self.gamma_t,
            bias=self.beta_t,
            memory_config=memory_config,
            core_grid=self.core_grid,
            epsilon=self.norm_eps,
            inplace=inplace,
            num_out_blocks=num_out_blocks,
        )

        # Convert back to DRAM if sharded
        if self.sharded:
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)

        return hidden_states

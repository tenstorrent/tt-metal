# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn
from ..parallel_config import VAEParallelConfig


if TYPE_CHECKING:
    pass


# TODO: Pass in dtype. Parameterize hardcoded values. Add epsilon as parameter so we can just from torch groupnorm
@dataclass
class TtGroupNormParameters:
    weight: ttnn.Tensor
    bias: ttnn.Tensor
    mask: ttnn.Tensor
    num_channels: int
    num_groups: int
    eps: float
    core_grid: ttnn.CoreGrid
    parallel_config: VAEParallelConfig

    @classmethod
    def from_torch(
        cls,
        torch_groupnorm,
        *,
        parallel_config: VAEParallelConfig,
    ) -> TtGroupNormParameters:
        num_channels = torch_groupnorm.num_channels
        num_groups = torch_groupnorm.num_groups

        # update core_grid.y to be a multiple of 32 as per group norm
        # Group norm produces wrong results if core_grid.x != core_grid.y. Observed with Chnls=128 and Grp=32
        assert (
            num_channels % 32 == 0 == num_channels % num_groups
        ), f"Incompatible channels ({num_channels}%32)or groups({num_channels}%{num_groups}):"

        opt_core_grid = parallel_config.device.core_grid
        grid_e = min(opt_core_grid.x, opt_core_grid.y)
        while num_channels % (32 * grid_e) != 0:
            grid_e -= 1

        opt_core_grid = ttnn.CoreGrid(y=grid_e, x=grid_e)  # Non uniform core grid causing issues with PCC

        torch_weight = ttnn.create_group_norm_weight_bias_rm(
            torch_groupnorm.state_dict()["weight"], num_channels, opt_core_grid.y
        )
        torch_bias = ttnn.create_group_norm_weight_bias_rm(
            torch_groupnorm.state_dict()["bias"], num_channels, opt_core_grid.y
        )

        torch_mask = ttnn.create_group_norm_input_mask(num_channels, num_groups, opt_core_grid.y)

        memory_config = ttnn.DRAM_MEMORY_CONFIG
        return cls(
            weight=ttnn.from_torch(
                torch_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=parallel_config.device,
                memory_config=memory_config,
            ),
            bias=ttnn.from_torch(
                torch_bias,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=parallel_config.device,
                memory_config=memory_config,
            ),
            mask=ttnn.from_torch(
                torch_mask,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=parallel_config.device,
                memory_config=memory_config,
            ),
            num_channels=num_channels,
            num_groups=num_groups,
            eps=torch_groupnorm.eps,
            core_grid=opt_core_grid,
            parallel_config=parallel_config,
        )


def vae_group_norm(x_in, parameters: TtGroupNormParameters):
    [batch_size, height, width, channels] = list(x_in.shape)

    # TODO: Compute optimal output blocks
    num_out_blocks = -(
        -width * height // (256 * parameters.core_grid.x * parameters.core_grid.y)
    )  # Prevents next step from hanging. TODO: Investigate
    x = ttnn.to_memory_config(x_in, ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = x.reshape([batch_size, 1, width * height, channels])
    x = ttnn.tilize_with_zero_padding(x, use_multicore=True)
    x = ttnn.group_norm(
        x,
        weight=parameters.weight,
        bias=parameters.bias,
        input_mask=parameters.mask,
        num_groups=parameters.num_groups,
        epsilon=parameters.eps,
        core_grid=parameters.core_grid,
        inplace=False,
        num_out_blocks=num_out_blocks,
        output_layout=ttnn.TILE_LAYOUT,
    )
    ttnn.synchronize_device(parameters.parallel_config.device)

    return x.reshape([batch_size, height, width, channels])

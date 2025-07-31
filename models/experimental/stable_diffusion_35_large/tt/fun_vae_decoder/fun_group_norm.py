# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import ttnn
from ..parallel_config import VAEParallelConfig

if TYPE_CHECKING:
    pass


# Assumptions: Output is always non sharded. Input sharding is expected to be configured by the caller.
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
    mesh_sharded_input: bool  # used to indicate the input is sharded. Ensure the tensor shape align for this
    allow_sharded_compute: bool  # used to indicate that output is sharded

    @classmethod
    def from_torch(
        cls,
        torch_groupnorm,
        *,
        parallel_config: VAEParallelConfig,
        mesh_sharded_input: bool = True,
        allow_sharded_compute: bool = True,  # override sharded compute by setting to false
    ) -> TtGroupNormParameters:
        num_channels = torch_groupnorm.num_channels
        num_groups = torch_groupnorm.num_groups
        mesh_shape = list(parallel_config.device.shape)
        device_count = mesh_shape[1]  # make this part of parallel config

        # Apply sharded config.
        # if allow_sharded_compute and mesh_sharded_input and ((num_channels % device_count) == 0 == (num_groups % device_count) == (num_channels / device_count) % 32):
        if mesh_sharded_input and allow_sharded_compute:  # TODO: Move to pweights prepare function
            num_channels = num_channels // device_count
            num_groups = num_groups // device_count

        # Group norm produces wrong results if core_grid.x != core_grid.y. Observed with Chnls=128 and Grp=32
        assert (
            num_channels % 32 == 0 == num_channels % num_groups
        ), f"Incompatible channels ({num_channels}%32)or groups({num_channels}%{num_groups}):"

        opt_core_grid = parallel_config.device.core_grid
        grid_e = min(opt_core_grid.x, opt_core_grid.y)
        while num_channels % (32 * grid_e) != 0:
            grid_e -= 1

        opt_core_grid = ttnn.CoreGrid(y=grid_e, x=grid_e)  # Non uniform core grid causing issues with PCC

        # torch_weight = ttnn.create_group_norm_weight_bias_rm(
        #     torch_groupnorm.state_dict()["weight"], num_channels, opt_core_grid.y
        # )
        # torch_bias = ttnn.create_group_norm_weight_bias_rm(
        #     torch_groupnorm.state_dict()["bias"], num_channels, opt_core_grid.y
        # )

        torch_weight, mesh_mapper_weight = group_norm_weight_bias_rm_sharded(
            torch_groupnorm.state_dict()["weight"],
            mesh_sharded_input,
            allow_sharded_compute,
            device_count,
            num_channels,
            opt_core_grid,
            parallel_config.device,
        )
        torch_bias, mesh_mapper_bias = group_norm_weight_bias_rm_sharded(
            torch_groupnorm.state_dict()["bias"],
            mesh_sharded_input,
            allow_sharded_compute,
            device_count,
            num_channels,
            opt_core_grid,
            parallel_config.device,
        )

        torch_mask = ttnn.create_group_norm_input_mask(num_channels, num_groups, opt_core_grid.y)

        memory_config = ttnn.DRAM_MEMORY_CONFIG
        return cls(
            weight=ttnn.from_torch(
                torch_weight,
                mesh_mapper=mesh_mapper_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=parallel_config.device,
                memory_config=memory_config,
            ),
            bias=ttnn.from_torch(
                torch_bias,
                mesh_mapper=mesh_mapper_bias,
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
            mesh_sharded_input=mesh_sharded_input,
            allow_sharded_compute=allow_sharded_compute,
        )


def group_norm_weight_bias_rm_sharded(
    tensor, mesh_sharded_input, allow_sharded_compute, device_count, num_channels, opt_core_grid, device
):
    if mesh_sharded_input and allow_sharded_compute:
        torch_sharded_lst = [
            ttnn.create_group_norm_weight_bias_rm(t, num_channels, opt_core_grid.y) for t in tensor.chunk(device_count)
        ]
        tensor_to_shard = torch.cat(torch_sharded_lst, dim=0)
        mesh_mapper = ttnn.ShardTensor2dMesh(device, tuple(device.shape), dims=[None, 0])
    else:
        tensor_to_shard = ttnn.create_group_norm_weight_bias_rm(tensor, num_channels, opt_core_grid.y)
        mesh_mapper = None

    return tensor_to_shard, mesh_mapper


# TODO: Move to parallel manager
def gn_all_gather(x, parameters: TtGroupNormParameters):
    x_g = ttnn.experimental.all_gather_async(
        input_tensor=x,
        dim=3,
        multi_device_global_semaphore=parameters.parallel_config.new_gather_handles,
        topology=ttnn.Topology.Linear,
        mesh_device=parameters.parallel_config.device,
        cluster_axis=1,
        num_links=1,
    )
    ttnn.synchronize_device(parameters.parallel_config.device)
    return x_g


def vae_group_norm(x, parameters: TtGroupNormParameters):
    batch_size, height, width, _ = x.shape
    channels = parameters.num_channels

    # TODO: Compute optimal output blocks
    num_out_blocks = -(
        -width * height // (256 * parameters.core_grid.x * parameters.core_grid.y)
    )  # Prevents next step from hanging. TODO: Investigate

    if not parameters.allow_sharded_compute and parameters.mesh_sharded_input:
        x = gn_all_gather(x, parameters)

    x = x.reshape([batch_size, 1, width * height, channels])
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
    x = x.reshape([batch_size, height, width, channels])
    if parameters.mesh_sharded_input and parameters.allow_sharded_compute:
        x = gn_all_gather(x, parameters)

    return x

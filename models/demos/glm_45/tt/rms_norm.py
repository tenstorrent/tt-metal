# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from torch import nn

import ttnn
from models.demos.glm_45.config import MeshConfig
from models.demos.glm_45.utils.general_utils import get_cache_file_name


class RMSNorm(nn.Module):
    def __init__(self, mesh_device, hf_config, state_dict, tensor_cache_path=None, mesh_config=None):
        super().__init__()
        torch_weight = state_dict["weight"]

        # Use MeshConfig for clean parallelization
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])
        self.is_distributed = False  # self.mesh_config.tp > 1
        # Use TILE layout with shape [1, hidden] to ensure valid ROW broadcast during scaling
        self.tt_weight = ttnn.as_tensor(
            torch_weight.reshape((1, -1)),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_config.shard_mapper(mesh_device, mesh_dims=(None, -2))
            if self.is_distributed
            else None,
        )

        self.eps = hf_config.rms_norm_eps
        self.mesh_device = mesh_device

    def forward(self, x):
        if self.is_distributed:
            activation_grid_bounding_box_size = x.memory_config().shard_spec.grid.bounding_box().grid_size()
            shard_height, shard_width = x.memory_config().shard_spec.shape
            program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=activation_grid_bounding_box_size,
                subblock_w=1,
                block_h=ttnn.core.divup(shard_height, ttnn.TILE_SIZE),
                block_w=ttnn.core.divup(shard_width, ttnn.TILE_SIZE),
                inplace=False,
            )
            # If the activation is sharded, we need to use an optimized rmsnorm

            tt_gathered_stats_memory_config = ttnn.create_sharded_memory_config(
                shape=[1, 1, 32, 32 * self.mesh_shape[1]],
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            # Run distributed rmsnorm part 1
            tt_stats = ttnn.rms_norm_pre_all_gather(x, program_config=program_config, dtype=ttnn.bfloat16)

            # AllGather stats
            tt_gathered_stats = ttnn.all_gather(
                tt_stats,
                dim=3,
                num_links=1,
                cluster_axis=1,
                mesh_device=self.mesh_device,
                memory_config=tt_gathered_stats_memory_config,
                topology=ttnn.Topology.Linear,
            )
            ttnn.deallocate(tt_stats)

            # Run distributed rmsnorm part 2
            tt_output = ttnn.rms_norm_post_all_gather(
                x,
                tt_gathered_stats,
                program_config=program_config,
                epsilon=self.eps,
                weight=self.tt_weight,
                dtype=ttnn.bfloat16,
                stats=tt_gathered_stats,
            )
            ttnn.deallocate(tt_gathered_stats)
            return tt_output
        else:
            tt_output = ttnn.rms_norm(
                x,
                weight=self.tt_weight,
                epsilon=self.eps,
                # program_config=program_config,
            )
            return tt_output

# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from torch import nn

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.utils.general_utils import get_cache_file_name


class RMSNorm(nn.Module):
    def __init__(self, mesh_device, hf_config, state_dict, tensor_cache_path=None, mesh_config=None, with_scale=True):
        super().__init__()
        self.with_scale = with_scale

        if with_scale and state_dict and "weight" in state_dict:
            torch_weight = state_dict["weight"].reshape((1, 1, -1, ttnn.TILE_SIZE))
        else:
            torch_weight = None

        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1]))
        self.is_distributed = False

        if with_scale:
            self.tt_weight = ttnn.as_tensor(
                torch_weight,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.mesh_config.shard_mapper(mesh_device, mesh_dims=(None, -2))
                if self.is_distributed
                else None,
            )
        else:
            self.tt_weight = None

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

            tt_gathered_stats_memory_config = ttnn.create_sharded_memory_config(
                shape=[1, 1, 32, 32 * self.mesh_shape[1]],
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            tt_stats = ttnn.rms_norm_pre_all_gather(x, program_config=program_config, dtype=ttnn.bfloat16)

            tt_gathered_stats = ttnn.all_gather(
                tt_stats,
                dim=3,
                num_links=1,
                cluster_axis=1,
                mesh_device=self.mesh_device,
                memory_config=tt_gathered_stats_memory_config,
                topology=ttnn.Topology.Ring,
            )
            ttnn.deallocate(tt_stats)

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
            if self.with_scale:
                tt_output = ttnn.rms_norm(
                    x,
                    weight=self.tt_weight,
                    epsilon=self.eps,
                )
            else:
                tt_output = ttnn.rms_norm(
                    x,
                    epsilon=self.eps,
                )
            return tt_output

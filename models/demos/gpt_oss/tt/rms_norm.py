# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

import ttnn
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name


class RMSNorm(nn.Module):
    def __init__(self, mesh_device, hf_config, state_dict, tensor_cache_path=None, mesh_config=None, ccl_manager=None):
        super().__init__()
        torch_weight = state_dict["weight"]

        # Use MeshConfig for clean parallelization
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1]))
        self.num_rows = self.mesh_config.mesh_shape[0]
        self.num_cols = self.mesh_config.mesh_shape[1]
        self.ccl_manager = ccl_manager

        # For 2D sharding: enable distributed norm when we have multiple rows (EP dimension)
        self.is_distributed = self.num_rows > 1

        # Pad weight if needed for distributed sharding
        hidden_size = hf_config.hidden_size
        if self.is_distributed:
            # Need to pad to be divisible by (num_rows * tile_size)
            shard_chunk_size = self.num_rows * ttnn.TILE_SIZE
            if hidden_size % shard_chunk_size != 0:
                padded_size = ((hidden_size + shard_chunk_size - 1) // shard_chunk_size) * shard_chunk_size
                padding_size = padded_size - hidden_size
                # Pad with ones (neutral element for RMSNorm scaling)
                torch_weight = torch.nn.functional.pad(torch_weight, (0, padding_size), value=1.0)

        self.tt_weight = ttnn.as_tensor(
            torch_weight.reshape((1, 1, -1, ttnn.TILE_SIZE)),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=list(mesh_device.shape))
            if self.is_distributed
            else None,
        )

        self.eps = hf_config.rms_norm_eps
        self.mesh_device = mesh_device

    def forward(self, x):
        if self.is_distributed:
            num_cores_ln = 16
            hidden_size_per_device_distributed_ln = 720
            self.gather_in_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(1, 1, 32, 32),  # hidden_size_per_device_distributed_ln//num_cores_ln),
                core_grid=ttnn.CoreGrid(y=3, x=8),
                strategy=ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )
            program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(8, 3),
                subblock_w=1,  # (hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
                block_h=1,
                block_w=1,  # (hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
                inplace=False,
            )
            x = ttnn.to_memory_config(x, self.gather_in_mem_cfg)
            print("x", x.memory_config())
            # For 2D sharding: input is row-sharded (hidden/num_rows per row)
            # Need to gather stats across ROWS (cluster_axis=0)
            # activation_grid_bounding_box_size = x.memory_config().shard_spec.grid.bounding_box().grid_size()
            # shard_height, shard_width = x.memory_config().shard_spec.shape

            tt_gathered_stats_memory_config = ttnn.create_sharded_memory_config(
                shape=[1, 1, 32, 32 * 4],  # Gather across rows
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.WIDTH,
                # orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            # Run distributed rmsnorm part 1
            x = ttnn.unsqueeze_to_4D(x)
            tt_stats = ttnn.rms_norm_pre_all_gather(x, program_config=program_config)
            print("done tt stats", tt_stats.memory_config())

            # AllGather stats across ROWS (cluster_axis=0) for row-sharded input
            tt_gathered_stats = ttnn.all_gather(
                tt_stats,
                dim=3,
                num_links=2,
                cluster_axis=0,  # Changed from 1 to 0 for row-sharding
                # mesh_device=self.mesh_device,
                memory_config=tt_gathered_stats_memory_config,
                # ccl_manager = self.ccl_manager
                topology=ttnn.Topology.Linear,
            )
            print("done ag", tt_gathered_stats.memory_config())
            ttnn.deallocate(tt_stats)
            # Run distributed rmsnorm part 2
            tt_output = ttnn.rms_norm_post_all_gather(
                x,
                tt_gathered_stats,
                program_config=program_config,
                epsilon=self.eps,
                weight=self.tt_weight,
                dtype=ttnn.bfloat8_b,
                # stats=tt_gathered_stats,
                # memory_config= ttnn.L1_MEMORY_CONFIG
            )
            ttnn.deallocate(tt_gathered_stats)
            print("tt_output", tt_output.shape)
            tt_output = ttnn.reshape(tt_output, (1, tt_output.shape[-2], 736))
            # tt_output = ttnn.to_torc
            # tt_output = ttnn.all_gather(tt_output, cluster_axis=0, dim=-1)
            return tt_output
        else:
            # Non-distributed: simple RMSNorm
            tt_output = ttnn.rms_norm(
                x,
                weight=self.tt_weight,
                epsilon=self.eps,
            )
            return tt_output

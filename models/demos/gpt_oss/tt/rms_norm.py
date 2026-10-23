# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from torch import nn

import ttnn
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name, get_default_num_links

from . import opt_switches


class RMSNorm(nn.Module):
    def __init__(self, mesh_device, hf_config, state_dict, tensor_cache_path=None, mesh_config=None):
        super().__init__()
        if state_dict:
            torch_weight = state_dict["weight"].reshape((1, 1, -1, ttnn.TILE_SIZE))
        else:
            torch_weight = None

        # Use MeshConfig for clean parallelization
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1]))
        self.is_distributed = False  # self.mesh_config.tp > 1
        self.tt_weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=(
                self.mesh_config.shard_mapper(mesh_device, mesh_dims=(None, -2)) if self.is_distributed else None
            ),
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
                num_links=get_default_num_links(self.mesh_device),
                cluster_axis=1,
                mesh_device=self.mesh_device,
                memory_config=tt_gathered_stats_memory_config,
                topology=ttnn.Topology.Ring,
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
            # GPT_OSS_OPT_DECODE_NORM: multi-core sharded RMSNorm for one-tile-row inputs.
            if (
                opt_switches.on("DECODE_NORM")
                and not x.is_sharded()
                and x.shape[-2] == 1
                and x.padded_shape[-2] == ttnn.TILE_SIZE
            ):
                return self._forward_one_tile_row_sharded(x)
            tt_output = ttnn.rms_norm(
                x,
                weight=self.tt_weight,
                epsilon=self.eps,
                # program_config=program_config,
            )
            return tt_output

    def _forward_one_tile_row_sharded(self, x):
        """GPT_OSS_OPT_DECODE_NORM: a one-token activation (decode) normed on 30
        cores (width-sharded [32, hidden / 30], sharded multi-core rms_norm as batched decode uses) instead of the
        interleaved rms_norm, which runs it on one core; returned in the input's memory config."""
        from .batched_decode import sharded_norm_program_config
        from .fused_decode.config import RESIDUAL_CORES, width_sharded_memory_config

        if x.shape[-1] % (RESIDUAL_CORES * ttnn.TILE_SIZE) != 0:
            return ttnn.rms_norm(x, weight=self.tt_weight, epsilon=self.eps)
        sharded = width_sharded_memory_config(self.mesh_device, x.shape[-1], RESIDUAL_CORES)
        xs = ttnn.to_memory_config(x, sharded)
        ys = ttnn.rms_norm(
            xs,
            weight=self.tt_weight,
            epsilon=self.eps,
            program_config=sharded_norm_program_config(sharded),
            memory_config=sharded,
        )
        xs.deallocate(True)
        y = ttnn.to_memory_config(ys, x.memory_config())
        ys.deallocate(True)
        return y

    def forward_sharded(self, x):
        """Batched decode residual stream (batched_decode.py): width-sharded multi-core norm on the activation's own
        shards."""
        from .batched_decode import sharded_norm_program_config

        return ttnn.rms_norm(
            x,
            weight=self.tt_weight,
            epsilon=self.eps,
            program_config=sharded_norm_program_config(x.memory_config()),
            memory_config=x.memory_config(),
        )

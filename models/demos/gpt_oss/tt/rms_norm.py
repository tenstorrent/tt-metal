# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from torch import nn

import ttnn
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name, get_default_num_links


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
            # Decode single-row case ([1,1,1,W] padded to [.,.,32,W]): the default
            # interleaved path auto-selects a SINGLE core to reduce all W/32 tiles
            # (~40us, 3% BW). Width-shard across cores + sharded program config
            # parallelizes the reduction (swept best = 9 cores for W=2880). The
            # shard HEIGHT must be the padded tile height (32), not the logical H=1.
            W = x.shape[-1]
            H = x.shape[-2]
            TILE = ttnn.TILE_SIZE
            Wt = W // TILE
            Ht = max(1, -(-H // TILE))  # ceil rows in tiles (H may be 1 at decode)
            ncores = 9 if Wt % 9 == 0 else next((c for c in (10, 6, 5, 3) if Wt % c == 0), 1)
            # Width-shard the norm across cores + sharded program config to parallelize
            # the reduction (1-core interleaved default ~40us/op at ~3% BW). Also moves
            # the tensor DRAM->L1 (higher aggregate BW). Wrapped in try/except: sharded
            # program build can TT_THROW on shapes not seen at development time (e.g.
            # multi-tile-row prefill norms), so fall back to the safe default path.
            if (
                x.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
                and ncores > 1
                and W % TILE == 0
                and Ht <= 4  # cap L1 shard footprint; larger seqs use default path
            ):
                try:
                    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ncores - 1, 0))})
                    # shard shape: padded tile-row height, width split across cores
                    shard_spec = ttnn.ShardSpec(grid, [Ht * TILE, W // ncores], ttnn.ShardOrientation.ROW_MAJOR)
                    sharded_mem = ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec
                    )
                    x_sh = ttnn.to_memory_config(x, sharded_mem)
                    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(ncores, 1),
                        subblock_w=1,
                        block_h=Ht,
                        block_w=Wt // ncores,
                        inplace=False,
                    )
                    out_sh = ttnn.rms_norm(
                        x_sh,
                        weight=self.tt_weight,
                        epsilon=self.eps,
                        program_config=pc,
                        memory_config=sharded_mem,
                    )
                    x_sh.deallocate(True)
                    tt_output = ttnn.sharded_to_interleaved(out_sh, ttnn.DRAM_MEMORY_CONFIG)
                    out_sh.deallocate(True)
                    return tt_output
                except Exception:
                    pass  # fall through to the robust default path
            tt_output = ttnn.rms_norm(
                x,
                weight=self.tt_weight,
                epsilon=self.eps,
            )
            return tt_output

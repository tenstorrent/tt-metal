# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from torch import nn

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.utils.general_utils import get_cache_file_name

# Row cutoff for the width-sharded fast path. The per-core input shard is
# (height, dim/num_cores) elements resident in L1 alongside other live
# buffers; at hidden=5376 / TP=8 a seq=4096 prefill norm needs ~786 KB/bank
# and TT_FATALs (Out of Memory) with only ~607 KB free. seq=1024 fits
# comfortably (~197 KB/bank). Longer prefill keeps the plain interleaved path.
_SHARDED_NORM_MAX_HEIGHT = 1024


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

        # Width-sharded fast path (decode AND prefill). The plain (interleaved)
        # rms_norm runs the RMS reduction over the full hidden width on few
        # cores — e.g. ~90 us for a [1,1,128,hidden] prefill norm on Gemma4-31B
        # (hidden=5376) landing on only 4 of 64 cores, since the interleaved op
        # parallelizes over row-tiles, not the hidden width. Width-sharding the
        # activation across a core grid parallelizes the reduction
        # (LayerNormShardedMultiCoreProgramConfig handles the cross-core gather)
        # regardless of how many row-tiles (``block_h``) the activation has.
        # Built lazily on first call at a given (hidden width, row count) shape
        # so we can read the activation's true (padded) dims, then cached.
        self._sharded_cfg = None  # (input_memcfg, program_config) or None if unavailable
        self._sharded_dim = None
        self._sharded_height = None

    def _build_sharded_cfg(self, dim, height):
        """Pick the largest core grid whose core count divides dim/32 and build
        the width-sharded input memcfg + LayerNorm program config for an
        activation of ``height`` rows (a multiple of TILE_SIZE — 32 for a
        decode/small-batch tile, or the prefill seq length otherwise). Returns
        None if no usable grid divides the tile-width evenly (falls back to
        plain)."""
        if dim % ttnn.TILE_SIZE != 0:
            return None
        tiles = dim // ttnn.TILE_SIZE
        grid = self.mesh_device.compute_with_storage_grid_size()
        best = None  # (num_cores, gx, gy)
        for gy in range(1, grid.y + 1):
            for gx in range(1, grid.x + 1):
                n = gx * gy
                if tiles % n == 0 and (best is None or n > best[0]):
                    best = (n, gx, gy)
        if best is None or best[0] == 1:
            return None
        num_cores, gx, gy = best
        block_w = tiles // num_cores
        subblock_w = 4
        while subblock_w > 1 and block_w % subblock_w != 0:
            subblock_w -= 1
        input_memcfg = ttnn.create_sharded_memory_config(
            shape=(height, dim // num_cores),
            core_grid=ttnn.CoreGrid(x=gx, y=gy),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[gx, gy],
            subblock_w=subblock_w,
            block_h=height // ttnn.TILE_SIZE,
            block_w=block_w,
            inplace=False,
        )
        return (input_memcfg, program_config)

    def _forward_sharded(self, x):
        """Width-sharded decode RMSNorm: I2S -> sharded rms_norm -> S2I."""
        x_sh = ttnn.to_memory_config(x, self._sharded_cfg[0])
        out = ttnn.rms_norm(
            x_sh,
            weight=self.tt_weight,
            epsilon=self.eps,
            program_config=self._sharded_cfg[1],
        )
        x_sh.deallocate(True)
        out_interleaved = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        out.deallocate(True)
        return out_interleaved

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

            # Avoid deprecated all_gather kwargs (num_links/topology/mesh_device).
            tt_gathered_stats = ttnn.all_gather(
                tt_stats,
                dim=3,
                cluster_axis=1,
                memory_config=tt_gathered_stats_memory_config,
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
            # Width-sharded fast path: a tile-aligned-height activation with a
            # learned weight and an interleaved layout → width-sharded rms_norm.
            # Covers both decode (height <= 32, padded to one tile) and prefill
            # (height a multiple of 32). The no-weight per-head norms keep the
            # plain path. Sharded config is (width, height)-specific, so rebuild
            # if either ever changes.
            padded_height = ((int(x.shape[-2]) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
            if (
                self.with_scale
                and self.tt_weight is not None
                and len(x.shape) == 4
                and 1 <= padded_height <= _SHARDED_NORM_MAX_HEIGHT
                and not x.is_sharded()
            ):
                dim = x.shape[-1]
                if self._sharded_cfg is None or self._sharded_dim != dim or self._sharded_height != padded_height:
                    self._sharded_dim = dim
                    self._sharded_height = padded_height
                    self._sharded_cfg = self._build_sharded_cfg(dim, padded_height)
                if self._sharded_cfg:
                    return self._forward_sharded(x)

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

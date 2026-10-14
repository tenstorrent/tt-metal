# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import functools

from torch import nn

import ttnn
from models.demos.gemma4_d_p.utils.general_utils import get_cache_file_name

# Block-shard geometry: core columns split the width, up to 8 rows of cores split the rows.
_MAX_GRID_Y = 8
# Larger per-core blocks fail to allocate the op's circular buffers in L1 (measured).
_MAX_BLOCK_TILES = 84


@functools.cache
def _block_shard_geometry(rows, width):
    """(block_h, block_w, grid_y, grid_x) for block-sharding a rows x width norm, or None for the default path.

    Interleaved ttnn.rms_norm uses one core per tile row whatever the width; block sharding also splits the width
    across core columns.
    """
    tile = ttnn.TILE_SIZE
    # Up to 512 rows, 12 columns with 2-row blocks measured 2.5x faster (256 rows: 25 -> 10 us). At 1024 rows
    # 12 columns was slower in the model, so taller slabs keep 8 columns with 4-row blocks.
    grid_x, min_block_h = (12, 2) if rows <= 512 else (8, 4)
    if rows % tile or width % tile or (width // tile) % grid_x:
        return None
    rows_t, block_w = rows // tile, width // tile // grid_x
    block_h = max(min_block_h, ttnn.core.divup(rows_t, _MAX_GRID_Y))
    if rows_t % block_h or block_h * block_w > _MAX_BLOCK_TILES:
        return None
    return block_h, block_w, rows_t // block_h, grid_x


@functools.cache
def _block_sharded_memory_config(block_h, block_w, grid_y, grid_x):
    tile = ttnn.TILE_SIZE
    return ttnn.create_sharded_memory_config(
        shape=(block_h * tile, block_w * tile),
        core_grid=ttnn.CoreGrid(x=grid_x, y=grid_y),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


@functools.cache
def _block_sharded_program_config(block_h, block_w, grid_y, grid_x):
    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[grid_x, grid_y],
        subblock_w=next(s for s in (3, 2, 1) if block_w % s == 0),
        block_h=block_h,
        block_w=block_w,
        inplace=False,
    )


class RMSNorm(nn.Module):
    def __init__(self, mesh_config, hf_config, state_dict, tensor_cache_path=None, with_scale=True):
        mesh_device = mesh_config.device
        super().__init__()
        self.with_scale = with_scale

        if with_scale and state_dict and "weight" in state_dict:
            torch_weight = state_dict["weight"].reshape((1, 1, -1, ttnn.TILE_SIZE))
        else:
            torch_weight = None

        self.mesh_config = mesh_config

        if with_scale:
            self.tt_weight = ttnn.as_tensor(
                torch_weight,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # TILE gamma enables the large-tensor RMSNorm path, which bounds L1 usage
            # for the FP32 intermediate buffers in the 5376-wide prefill norm.
            flat_weight = ttnn.reshape(self.tt_weight, (1, 1, 1, hf_config.hidden_size))
            self.tt_weight = ttnn.to_layout(flat_weight, ttnn.TILE_LAYOUT)
        else:
            self.tt_weight = None

        self.eps = hf_config.rms_norm_eps
        self.mesh_device = mesh_device
        # FP32 accumulation as in the reference's FP32 norm; HiFi2 multiplies keep the 256k KV-cache PCC gate.
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def shard_memory_config(self, x):
        """The block-sharded layout this norm runs in for x's shape, or None."""
        geometry = _block_shard_geometry(x.padded_shape[-2], x.padded_shape[-1])
        return None if geometry is None else _block_sharded_memory_config(*geometry)

    def forward(self, x, memory_config=None):
        """A memory_config equal to shard_memory_config(x) returns the block-sharded output as is; an x already in
        that layout is not resharded."""
        geometry = _block_shard_geometry(x.padded_shape[-2], x.padded_shape[-1])
        if geometry is None:
            return ttnn.rms_norm(
                x,
                weight=self.tt_weight,
                epsilon=self.eps,
                memory_config=memory_config,
                compute_kernel_config=self.compute_kernel_config,
            )
        shard_memory_config = _block_sharded_memory_config(*geometry)
        program_config = _block_sharded_program_config(*geometry)
        x_sharded = x if x.memory_config() == shard_memory_config else ttnn.to_memory_config(x, shard_memory_config)
        out_sharded = ttnn.rms_norm(
            x_sharded,
            weight=self.tt_weight,
            epsilon=self.eps,
            program_config=program_config,
            memory_config=shard_memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        if x_sharded is not x:
            x_sharded.deallocate(True)
        if memory_config is not None and memory_config == shard_memory_config:
            return out_sharded
        out = ttnn.sharded_to_interleaved(out_sharded, memory_config or ttnn.DRAM_MEMORY_CONFIG)
        out_sharded.deallocate(True)
        return out

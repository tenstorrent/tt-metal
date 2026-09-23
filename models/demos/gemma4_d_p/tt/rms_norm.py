# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import functools

from torch import nn

import ttnn
from models.demos.gemma4_d_p.utils.general_utils import get_cache_file_name

# Sharded-norm geometry: 8 core columns split the width, up to 8 rows of cores split the rows.
_GRID = 8
# Shorter per-core blocks measured slower despite putting more cores on the norm.
_MIN_BLOCK_H = 4
# Larger per-core blocks fail to allocate the op's circular buffers in L1 (measured).
_MAX_BLOCK_TILES = 84


@functools.cache
def _block_sharded_config(rows, width):
    """Block-sharded memory and program config for a rows x width norm, or None for the default path.

    Interleaved ttnn.rms_norm parallelises over rows only, one core per tile row, so a device's 8,
    16 or 32 tile rows run on as many cores whatever the width. Also splitting the width over 8 core
    columns puts 16, 32 or 64 cores on the same rows.
    """
    tile = ttnn.TILE_SIZE
    if rows % tile or width % tile or (width // tile) % _GRID:
        return None
    rows_t, block_w = rows // tile, width // tile // _GRID
    block_h = max(_MIN_BLOCK_H, ttnn.core.divup(rows_t, _GRID))
    if rows_t % block_h or block_h * block_w > _MAX_BLOCK_TILES:
        return None
    grid_y = rows_t // block_h
    memory_config = ttnn.create_sharded_memory_config(
        shape=(block_h * tile, block_w * tile),
        core_grid=ttnn.CoreGrid(x=_GRID, y=grid_y),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[_GRID, grid_y],
        subblock_w=next(s for s in (3, 2, 1) if block_w % s == 0),
        block_h=block_h,
        block_w=block_w,
        inplace=False,
    )
    return memory_config, program_config


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
        # Match the reference's FP32 prefill RMSNorm computation.
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def forward(self, x, memory_config=None):
        sharded = _block_sharded_config(x.padded_shape[-2], x.padded_shape[-1])
        if sharded is None:
            return ttnn.rms_norm(
                x,
                weight=self.tt_weight,
                epsilon=self.eps,
                memory_config=memory_config,
                compute_kernel_config=self.compute_kernel_config,
            )
        shard_memory_config, program_config = sharded
        x_sharded = ttnn.to_memory_config(x, shard_memory_config)
        out_sharded = ttnn.rms_norm(
            x_sharded,
            weight=self.tt_weight,
            epsilon=self.eps,
            program_config=program_config,
            memory_config=shard_memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        x_sharded.deallocate(True)
        out = ttnn.sharded_to_interleaved(out_sharded, memory_config or ttnn.DRAM_MEMORY_CONFIG)
        out_sharded.deallocate(True)
        return out

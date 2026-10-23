# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 RMSNorm (y = x * rsqrt(mean(x^2) + eps) * w, no 1 + w) on the residual's TP row split.

Adapted from models/demos/gemma4_d_p/tt/rms_norm.py. Each chip holds [1, 1, rows, 5376] (160 rows between blocks: 640
SP rows split over the 4 TP columns), so the norm needs no collective. The weight is bf16, replicated on every chip.
NORM_WEIGHT fp32 (default): the unweighted norm in fp32, then the weight multiply to bf16 (see the setting).
"""

import functools

import torch

import ttnn

from .settings import settings

# Block-shard geometry: core columns split the width, up to 8 rows of cores split the rows.
_MAX_GRID_Y = 8
# Larger per-core blocks fail to allocate the op's circular buffers in L1 (gemma4_d_p, measured).
_MAX_BLOCK_TILES = 84


@functools.cache
def _block_shard_geometry(rows, width):
    """(block_h, block_w, grid_y, grid_x) for block-sharding a rows x width norm, or None for the interleaved path."""
    tile = ttnn.TILE_SIZE
    grid_x, min_block_h = (12, 2) if rows <= 512 else (8, 4)
    if rows % tile or width % tile or (width // tile) % grid_x:
        return None
    rows_t, block_w = rows // tile, width // tile // grid_x
    block_h = max(min_block_h, ttnn.core.divup(rows_t, _MAX_GRID_Y))
    if rows_t % block_h:
        block_h = next(h for h in range(ttnn.core.divup(rows_t, _MAX_GRID_Y), rows_t + 1) if rows_t % h == 0)
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


@functools.cache
def _fp32_shard_geometry(rows, width):
    """(block_h, block_w, grid_y, grid_x) for the fp32 norm (NORM_FP32_SHARDED): one tile row per core row, the width
    over 12 (or 8) core columns; None when the shape does not split so."""
    tile = ttnn.TILE_SIZE
    if rows % tile or width % tile or rows // tile > 10:
        return None
    w_t = width // tile
    grid_x = next((g for g in (12, 8, 6, 4) if w_t % g == 0), None)
    if grid_x is None:
        return None
    return 1, w_t // grid_x, rows // tile, grid_x


class RMSNorm:
    def __init__(self, mesh, weight: torch.Tensor | None, eps: float):
        """weight: [hidden] (None: unscaled norm). Uploaded once, replicated over the mesh."""
        self.eps = eps
        self.tt_weight = None
        if weight is not None:
            hidden = weight.numel()
            # ROW_MAJOR [.., hidden/32, 32] then TILE [1, 1, 1, hidden]: the TILE gamma takes rms_norm's large-tensor
            # path, which bounds L1 for the fp32 intermediates of the 5376-wide norm (gemma4_d_p).
            w = ttnn.from_torch(
                weight.reshape(1, 1, -1, ttnn.TILE_SIZE).to(torch.bfloat16),
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            self.tt_weight = ttnn.to_layout(ttnn.reshape(w, (1, 1, 1, hidden)), ttnn.TILE_LAYOUT)
        self.sharded = settings.get("NORM_SHARDED")
        self.fp32 = self.tt_weight is not None and settings.get("NORM_WEIGHT") == "fp32"
        self.fp32_gamma = self.tt_weight is not None and settings.get("NORM_WEIGHT") == "fp32_gamma"
        self.fp32_sharded = settings.get("NORM_FP32_SHARDED")
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, settings.get("NORM_FIDELITY")),
            math_approx_mode=settings.get("MATH_APPROX_MODE"),
            fp32_dest_acc_en=settings.get("NORM_FP32_ACC"),
            packer_l1_acc=settings.get("PACKER_L1_ACC"),
        )

    def __call__(self, x, memory_config=None, out_dtype=ttnn.bfloat16):
        """x: [1, 1, rows, hidden] bf16 (or fp32: the device model's residual stream) TILE on every chip. Returns the same shape, interleaved in memory_config (DRAM by
        default)."""
        if self.fp32:
            x32 = ttnn.typecast(x, ttnn.float32) if x.dtype != ttnn.float32 else x
            geometry = _fp32_shard_geometry(x32.padded_shape[-2], x32.padded_shape[-1]) if self.fp32_sharded else None
            if geometry is None:
                y = ttnn.rms_norm(
                    x32,
                    epsilon=self.eps,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    compute_kernel_config=self.compute_kernel_config,
                )
            else:
                # One tile row per core row: the interleaved norm runs one core per tile row (5 cores for 160 rows).
                shard_mc = _block_sharded_memory_config(*geometry)
                xs = ttnn.to_memory_config(x32, shard_mc)
                ys = ttnn.rms_norm(
                    xs,
                    epsilon=self.eps,
                    program_config=_block_sharded_program_config(*geometry),
                    memory_config=shard_mc,
                    compute_kernel_config=self.compute_kernel_config,
                )
                ttnn.deallocate(xs)
                y = ttnn.sharded_to_interleaved(ys, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(ys)
            if x32 is not x:
                ttnn.deallocate(x32)
            out = ttnn.multiply(
                y, self.tt_weight, dtype=ttnn.bfloat16, memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(y)
            return out
        if self.fp32_gamma:
            return self._fp32_gamma(x, memory_config or ttnn.DRAM_MEMORY_CONFIG, out_dtype)
        geometry = _block_shard_geometry(x.padded_shape[-2], x.padded_shape[-1]) if self.sharded else None
        if geometry is None:
            return ttnn.rms_norm(
                x,
                weight=self.tt_weight,
                epsilon=self.eps,
                memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
            )
        shard_mc = _block_sharded_memory_config(*geometry)
        x_sharded = ttnn.to_memory_config(x, shard_mc)
        out_sharded = ttnn.rms_norm(
            x_sharded,
            weight=self.tt_weight,
            epsilon=self.eps,
            program_config=_block_sharded_program_config(*geometry),
            memory_config=shard_mc,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(x_sharded)
        out = ttnn.sharded_to_interleaved(out_sharded, memory_config or ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out_sharded)
        return out

    def _fp32_gamma(self, x, memory_config, out_dtype):
        """NORM_WEIGHT=fp32_gamma: one rms_norm on the fp32 input with the weight as its gamma (fp32 out, fp32
        accumulation), block-sharded one tile row per core row when NORM_FP32_SHARDED; the output in out_dtype (the
        sharded path converts while unsharding). Post norms keep fp32 for the fp32 residual add (no bf16 rounding)."""
        x32 = ttnn.typecast(x, ttnn.float32) if x.dtype != ttnn.float32 else x
        geometry = _fp32_shard_geometry(x32.padded_shape[-2], x32.padded_shape[-1]) if self.fp32_sharded else None
        if geometry is None:
            y = ttnn.rms_norm(
                x32,
                weight=self.tt_weight,
                epsilon=self.eps,
                memory_config=memory_config,
                compute_kernel_config=self.compute_kernel_config,
            )
            if out_dtype != ttnn.float32:
                yd = ttnn.typecast(y, out_dtype)
                ttnn.deallocate(y)
                y = yd
        else:
            shard_mc = _block_sharded_memory_config(*geometry)
            xs = ttnn.to_memory_config(x32, shard_mc)
            ys = ttnn.rms_norm(
                xs,
                weight=self.tt_weight,
                epsilon=self.eps,
                program_config=_block_sharded_program_config(*geometry),
                memory_config=shard_mc,
                compute_kernel_config=self.compute_kernel_config,
            )
            ttnn.deallocate(xs)
            y = ttnn.sharded_to_interleaved(ys, memory_config, output_dtype=out_dtype)
            ttnn.deallocate(ys)
        if x32 is not x:
            ttnn.deallocate(x32)
        return y

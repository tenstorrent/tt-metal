# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Activation memory placement for dots.ocr: interleaved L1 when small, else DRAM.

Height-sharded L1 can be valid for *inputs* to matmul/linear when shard shapes satisfy TILE;
many ops still reject height-sharded *outputs*. Prefer interleaved L1 or DRAM for outputs.
"""

from __future__ import annotations

import ttnn

# bf16 element count above which we use DRAM instead of interleaved L1 (~256 KiB).
DOTS_L1_MAX_ELEMENTS = 131072

# ``paged_update_cache`` hard-requires ``update_idxs_tensor`` to be in DRAM
# (Metal buffer type assert in paged_update_cache_device_operation.cpp).
DOTS_PAGED_UPDATE_IDX_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG


def dots_memory_config_from_num_elements(num_elements: int) -> ttnn.MemoryConfig:
    if num_elements <= DOTS_L1_MAX_ELEMENTS:
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG


def dots_activation_memory_config(tensor: ttnn.Tensor) -> ttnn.MemoryConfig:
    n = 1
    for d in tensor.shape:
        n *= int(d)
    return dots_memory_config_from_num_elements(n)


def dots_memory_config_for_volume(num_elements: int, mesh_device: ttnn.MeshDevice) -> ttnn.MemoryConfig:
    """Host-side volume (``from_torch``, RoPE tables, masks): interleaved L1 or DRAM.

    ``mesh_device`` is unused but kept for call-site compatibility.
    """
    _ = mesh_device
    return dots_memory_config_from_num_elements(num_elements)


def dots_linear_output_memory_config(num_output_elements: int) -> ttnn.MemoryConfig:
    """Output buffer for matmul / linear / embedding / ``nlp_create_qkv_heads`` (no shard)."""
    return dots_memory_config_from_num_elements(num_output_elements)


def dots_max_tile_rows_for_l1_input(k_features: int) -> int:
    """Tile-aligned max sequence rows with ``rows * K <= DOTS_L1_MAX_ELEMENTS`` (bf16 elements)."""
    tile = ttnn.TILE_SIZE
    k_dim = max(1, int(k_features))
    cap = DOTS_L1_MAX_ELEMENTS // k_dim
    return max(tile, (cap // tile) * tile)


def dots_try_height_shard_l1_input(device, x: ttnn.Tensor) -> ttnn.Tensor | None:
    """Stage TILE activations in height-sharded L1 (tile-aligned shards). Returns ``None`` if unsupported."""
    try:
        grid_sz = device.compute_with_storage_grid_size()
        core_grid = ttnn.CoreGrid(y=int(grid_sz.y), x=int(grid_sz.x))
        shape = tuple(int(d) for d in x.shape)
        mem = ttnn.create_sharded_memory_config_(
            shape,
            core_grid,
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
            tile_layout=True,
        )
        return ttnn.to_memory_config(x, mem)
    except (RuntimeError, ValueError):
        return None


def dots_stage_matmul_input_l1_or_shard(device, x: ttnn.Tensor) -> ttnn.Tensor:
    """Prefer interleaved L1 when ``rows*K`` fits; else height-sharded L1; else leave buffer (typically DRAM)."""
    m = int(x.shape[-2])
    k = int(x.shape[-1])
    if m * k <= DOTS_L1_MAX_ELEMENTS:
        try:
            return ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        except (RuntimeError, ValueError):
            pass
    sharded = dots_try_height_shard_l1_input(device, x)
    return sharded if sharded is not None else x


def dots_chunk_matmul_program_config_2d(
    device, *, m: int, k: int, n: int
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    """2D multicast matmul — wide ``N`` (e.g. lm_head vocab slice, patch embed ``embed_dim``).

    ``per_core_M`` / ``per_core_N`` are in **output tiles** (Metal divides ``Mt``, ``Nt`` by them).
    Use tile counts so ``num_blocks_* <= grid`` (see matmul_multicore_reuse_mcast_2d_program_factory).
    """
    tile = ttnn.TILE_SIZE
    grid_sz = device.compute_with_storage_grid_size()
    gx, gy = int(grid_sz.x), int(grid_sz.y)
    grid_size = (gx, gy)

    Mt = (int(m) + tile - 1) // tile
    Nt = (int(n) + tile - 1) // tile
    Kt = (int(k) + tile - 1) // tile

    in0_block_w = 1
    for w in range(min(8, Kt), 1, -1):
        if Kt % w == 0:
            in0_block_w = w
            break
    if in0_block_w == 1 and Kt > 1:
        in0_block_w = Kt

    # Minimal tile counts so blocks along x/y fit the compute grid (no transpose_mcast).
    per_core_m = max(1, (Mt + gy - 1) // gy)
    per_core_n = max(1, (Nt + gx - 1) // gx)

    max_subblock = 8
    sub_w_candidates = [i for i in range(1, max_subblock + 1) if per_core_n % i == 0]
    out_subblock_w = max(sub_w_candidates) if sub_w_candidates else 1
    sub_h_candidates = [
        i for i in range(1, max_subblock + 1) if per_core_m % i == 0 and i * out_subblock_w <= max_subblock
    ]
    out_subblock_h = max(sub_h_candidates) if sub_h_candidates else 1

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


def dots_chunk_matmul_program_config_1d(
    device, *, m: int, k: int, n: int, mcast_in0: bool = True
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """1D matmul with widened ``in0_block_w`` along K-tiles (chunked M).

    Validates ``num_blocks_y * num_blocks_x <= gx * gy`` (Matmul 1D factory).
    """
    tile_h = ttnn.TILE_SIZE
    tile_w = ttnn.TILE_SIZE
    grid_sz = device.compute_with_storage_grid_size()
    gx, gy = int(grid_sz.x), int(grid_sz.y)
    num_cores = gx * gy

    Mt = (int(m) + tile_h - 1) // tile_h
    Nt = (int(n) + tile_w - 1) // tile_w
    Kt = (int(k) + tile_w - 1) // tile_w

    in0_block_w = 1
    for w in range(min(8, Kt), 1, -1):
        if Kt % w == 0:
            in0_block_w = w
            break
    if in0_block_w == 1 and Kt > 1:
        in0_block_w = Kt

    # Prefer a single M-block (typical long-sequence / LM path); grow per_core_N until
    # num_blocks_y * num_blocks_x fits total cores (1D factory limit).
    per_core_M = Mt
    per_core_N = max(1, (Nt + num_cores - 1) // num_cores)
    num_blocks_y = (Mt - 1) // per_core_M + 1
    num_blocks_x = (Nt - 1) // per_core_N + 1
    while num_blocks_y * num_blocks_x > num_cores and per_core_N < Nt:
        per_core_N += 1
        num_blocks_x = (Nt - 1) // per_core_N + 1

    max_subblock = 8
    sub_w_candidates = [i for i in range(1, max_subblock + 1) if per_core_N % i == 0]
    out_subblock_w = max(sub_w_candidates) if sub_w_candidates else 1
    sub_h_candidates = [
        i for i in range(1, max_subblock + 1) if per_core_M % i == 0 and i * out_subblock_w <= max_subblock
    ]
    out_subblock_h = max(sub_h_candidates) if sub_h_candidates else 1

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=mcast_in0,
    )


def dots_row_major_height_sharded_l1(shape: tuple[int, ...], mesh_device: ttnn.MeshDevice) -> ttnn.MemoryConfig:
    """Height-sharded L1 for row-major staging tensors (e.g. embedding INT32 pad path).

    Device-side ``ttnn.pad`` on tiny interleaved-L1 buffers can clash with static
    circular-buffer regions on Wormhole; host-pad + sharded L1 re-upload avoids that.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)
    try:
        return ttnn.create_sharded_memory_config(
            tuple(int(x) for x in shape),
            core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
    except RuntimeError:
        return ttnn.DRAM_MEMORY_CONFIG


def dots_activation_l1_or_height_sharded_l1(tensor: ttnn.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.MemoryConfig:
    """Interleaved L1 if the tensor volume fits; else DRAM.

    Kept for API compatibility; ``mesh_device`` is ignored (height-sharded L1 output
    is not usable with many TTNN ops, including ``ttnn.linear``).
    """
    _ = mesh_device
    return dots_activation_memory_config(tensor)

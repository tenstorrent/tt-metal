# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# DRAM-sharded decode matmul helpers (WIDTH_SHARDED weights + L1 activations).

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch

import ttnn

TILE = ttnn.TILE_SIZE  # 32


def _gather_in0_dram_mm_program_config(
    *,
    grid: Any,
    m_seq: int,
    k_dim: int,
    n_dim: int,
    fuse_batch: bool,
    fused_activation: Optional[Any],
    compute_kernel_config: Any,
) -> Any:
    """gather_in0 matmul config for WIDTH_SHARDED L1 activations + DRAM weights."""
    num_cores = grid.x * grid.y
    if m_seq % TILE != 0:
        raise ValueError(f"gather_in0 linear expects M divisible by {TILE}, got m_seq={m_seq}")
    in0_block_w = k_dim // num_cores // TILE
    out_block_h = m_seq // TILE
    out_block_w = n_dim // num_cores // TILE
    num_blocks_y = (m_seq // TILE - 1) // out_block_h + 1
    num_blocks_x = (n_dim // TILE - 1) // out_block_w + 1
    if num_blocks_y * num_blocks_x != num_cores:
        raise ValueError(
            "gather_in0 ring tiling does not match core count: "
            f"m_seq={m_seq} k_dim={k_dim} n_dim={n_dim} num_cores={num_cores} "
            f"blocks_y={num_blocks_y} blocks_x={num_blocks_x}"
        )

    fp32 = getattr(compute_kernel_config, "fp32_dest_acc_en", False)
    max_subblock_w_h = 4 if fp32 else 8

    candidates_w = [i for i in range(1, max_subblock_w_h + 1) if out_block_w % i == 0]
    out_subblock_w = max(candidates_w) if candidates_w else 1
    candidates_h = [
        i for i in range(1, max_subblock_w_h + 1) if out_block_h % i == 0 and i * out_subblock_w <= max_subblock_w_h
    ]
    out_subblock_h = max(candidates_h) if candidates_h else 1

    hop_cores = ttnn.CoreRangeSet(set())
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fuse_batch=fuse_batch,
        fused_activation=fused_activation,
        mcast_in0=False,
        gather_in0=True,
        hop_cores=hop_cores,
        num_global_cb_receivers=1,
    )


def width_sharded_l1_linear_keep_sharded(
    configuration: Any,
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    *,
    m_seq: int,
    k_dim: int,
    n_dim: int,
    fuse_batch: bool,
    compute_kernel_config: Any,
    fused_activation: Optional[Any] = None,
) -> ttnn.Tensor:
    """Width-sharded L1 linear; returns WIDTH_SHARDED L1 (no untilize→tilize to interleaved)."""
    if k_dim % TILE != 0 or n_dim % TILE != 0:
        raise ValueError(
            f"width_sharded_l1_linear_keep_sharded requires tile-aligned k_dim, n_dim; got k={k_dim} n={n_dim}"
        )

    grid = configuration.dram_shard_core_grid_for_k_and_n(k_dim, n_dim)
    num_cores = grid.x * grid.y
    m_tiles = math.ceil(m_seq / TILE)
    k_tiles = k_dim // TILE
    n_tiles = n_dim // TILE

    if k_tiles % num_cores != 0 or n_tiles % num_cores != 0:
        raise ValueError(
            f"width_sharded_l1_linear_keep_sharded shard mismatch: m={m_seq} k={k_dim} n={n_dim} cores={num_cores}"
        )

    in_mem = width_sharded_l1_memcfg(m_tiles, k_tiles, grid.x, grid.y)
    if not x.is_sharded():
        x = ttnn.interleaved_to_sharded(x, in_mem)

    prog_cfg = _gather_in0_dram_mm_program_config(
        grid=grid,
        m_seq=m_seq,
        k_dim=k_dim,
        n_dim=n_dim,
        fuse_batch=fuse_batch,
        fused_activation=fused_activation,
        compute_kernel_config=compute_kernel_config,
    )
    out_mem = width_sharded_l1_memcfg(m_tiles, n_tiles, grid.x, grid.y)

    return ttnn.linear(
        x,
        weight,
        dtype=ttnn.bfloat16,
        memory_config=out_mem,
        compute_kernel_config=compute_kernel_config,
        program_config=prog_cfg,
    )


def width_sharded_l1_linear_reuse_multicast(
    configuration: Any,
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    *,
    m_seq: int,
    k_dim: int,
    n_dim: int,
    fuse_batch: bool,
    compute_kernel_config: Any,
    fused_activation: Optional[Any] = None,
) -> ttnn.Tensor:
    """Width-sharded L1 linear with gather_in0 (tile-aligned k, n, m)."""
    if k_dim % TILE != 0 or n_dim % TILE != 0:
        raise ValueError(
            f"width_sharded_l1_linear_reuse_multicast requires tile-aligned k_dim, n_dim; got k={k_dim} n={n_dim}"
        )

    grid = configuration.dram_shard_core_grid_for_k_and_n(k_dim, n_dim)
    num_cores = grid.x * grid.y
    m_tiles = math.ceil(m_seq / TILE)
    k_tiles = k_dim // TILE
    n_tiles = n_dim // TILE

    if k_tiles % num_cores != 0 or n_tiles % num_cores != 0:
        raise ValueError(
            f"width_sharded_l1_linear_reuse_multicast shard mismatch: m={m_seq} k={k_dim} n={n_dim} cores={num_cores}"
        )

    in_mem = width_sharded_l1_memcfg(m_tiles, k_tiles, grid.x, grid.y)
    x_sharded = ttnn.interleaved_to_sharded(x, in_mem)

    prog_cfg = _gather_in0_dram_mm_program_config(
        grid=grid,
        m_seq=m_seq,
        k_dim=k_dim,
        n_dim=n_dim,
        fuse_batch=fuse_batch,
        fused_activation=fused_activation,
        compute_kernel_config=compute_kernel_config,
    )
    out_mem = width_sharded_l1_memcfg(m_tiles, n_tiles, grid.x, grid.y)

    out = ttnn.linear(
        x_sharded,
        weight,
        dtype=ttnn.bfloat16,
        memory_config=out_mem,
        compute_kernel_config=compute_kernel_config,
        program_config=prog_cfg,
    )
    ttnn.deallocate(x_sharded)
    out_il = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(out)
    return out_il


def _largest_divisor(n: int, max_divisor: int = 8) -> int:
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


def find_grid_k_n(K_tiles: int, N_tiles: int, max_rows: int = 10, max_cols: int = 13) -> Tuple[int, int]:
    """Largest core grid (rows, cols) where num_cores divides BOTH K_tiles and N_tiles."""
    max_cores = max_rows * max_cols
    candidates = [c for c in range(1, max_cores + 1) if K_tiles % c == 0 and N_tiles % c == 0]
    candidates.sort(reverse=True)
    for cores in candidates:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols
    raise AssertionError(f"No grid divides both K={K_tiles} and N={N_tiles} tiles within {max_rows}x{max_cols}.")


def find_unified_grid_mlp(
    k_tiles_gu: int,
    n_tiles_gu: int,
    k_tiles_d: int,
    n_tiles_d: int,
    max_rows: int = 10,
    max_cols: int = 13,
) -> Tuple[int, int]:
    """Single decode DRAM grid for gate/up and down SwiGLU projections."""
    max_cores = max_rows * max_cols
    candidates = [
        c
        for c in range(1, max_cores + 1)
        if k_tiles_gu % c == 0 and n_tiles_gu % c == 0 and k_tiles_d % c == 0 and n_tiles_d % c == 0
    ]
    candidates.sort(reverse=True)
    for cores in candidates:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols
    raise AssertionError(
        f"No unified MLP grid for k_gu={k_tiles_gu}, n_gu={n_tiles_gu}, k_d={k_tiles_d}, n_d={n_tiles_d} tiles."
    )


def pad_n_for_dram_align(n: int, dram_cores: int) -> int:
    """Pad N up to a multiple of TILE*dram_cores so weight shards evenly across DRAM banks."""
    align = TILE * dram_cores
    return ((n + align - 1) // align) * align


def dram_sharded_weight_memcfg(shard_height: int, n_cols_on_device: int, device) -> ttnn.MemoryConfig:
    """WIDTH_SHARDED DRAM for one device's slice: ``n_cols_on_device`` already DRAM-aligned."""
    dram_grid_size = device.dram_grid_size()
    dram_cores = dram_grid_size.x
    assert dram_grid_size.y == 1, "DRAM-sharded weights assume y=1 dram grid"
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))})
    assert n_cols_on_device % dram_cores == 0, f"N={n_cols_on_device} not divisible by dram_cores={dram_cores}"
    shard_spec = ttnn.ShardSpec(
        dram_grid,
        (shard_height, n_cols_on_device // dram_cores),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def width_sharded_l1_memcfg(m_tiles: int, k_tiles: int, num_cores_x: int, num_cores_y: int) -> ttnn.MemoryConfig:
    """Width-shard activations [1,1,m*TILE, k*TILE] across the compute grid."""
    num_cores = num_cores_x * num_cores_y
    assert k_tiles % num_cores == 0, f"K_tiles={k_tiles} not divisible by num_cores={num_cores}"
    core_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}
    )
    shard_spec = ttnn.ShardSpec(
        core_grid,
        (m_tiles * TILE, (k_tiles // num_cores) * TILE),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)


def dram_sharded_program_config(
    m: int, k: int, n: int, num_cores: int, fused_activation=None
) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
    assert k % (TILE * num_cores) == 0, f"K={k} not divisible by TILE*num_cores={TILE * num_cores}"
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=_largest_divisor(k // (TILE * num_cores)),
        per_core_M=math.ceil(m / TILE),
        per_core_N=math.ceil(n / (TILE * num_cores)),
        fused_activation=fused_activation,
    )


def build_dram_sharded_weight(
    weight_torch_2d: torch.Tensor,
    device,
    dtype=ttnn.bfloat16,
    mesh_mapper=None,
) -> Tuple[ttnn.Tensor, int, int]:
    """Build DRAM width-sharded weight [K,N]; returns (tensor, k, n_padded)."""
    assert weight_torch_2d.dim() == 2
    k, n_unpadded = weight_torch_2d.shape
    dram_cores = device.dram_grid_size().x
    n_padded = pad_n_for_dram_align(n_unpadded, dram_cores)
    if n_padded != n_unpadded:
        pad_w = torch.zeros(k, n_padded - n_unpadded, dtype=weight_torch_2d.dtype)
        weight_torch_2d = torch.cat([weight_torch_2d, pad_w], dim=1).contiguous()
    weight_4d = weight_torch_2d.unsqueeze(0).unsqueeze(0).contiguous()
    memcfg = dram_sharded_weight_memcfg(k, n_padded, device)
    tt = ttnn.from_torch(
        weight_4d,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memcfg,
        mesh_mapper=mesh_mapper,
    )
    return tt, k, n_padded


__all__ = [
    "TILE",
    "build_dram_sharded_weight",
    "dram_sharded_program_config",
    "dram_sharded_weight_memcfg",
    "find_grid_k_n",
    "find_unified_grid_mlp",
    "pad_n_for_dram_align",
    "width_sharded_l1_linear_keep_sharded",
    "width_sharded_l1_linear_reuse_multicast",
    "width_sharded_l1_memcfg",
]

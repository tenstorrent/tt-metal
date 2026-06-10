# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DRAM-sharded matmul / sharded-RMSNorm config helpers for the dots.ocr decode path.

These mirror the tt_transformers ``model_config.py`` helpers
(``create_dram_sharded_mem_config``, ``dram_matmul_config``,
``find_grid_k_n``, ``create_sharded_norm_config``) but are self-contained for the
dots.ocr single-device (p150 / Blackhole) decode path, where the full
``ModelArgs`` plumbing is not available.

Used ONLY by the sharded-decode flow (``sharded_decode=True``). The default
(interleaved / bf8) decode and the whole prefill path are untouched.

p150 has 8 DRAM banks (``dram_grid_size().x == 8``). DRAM-sharded matmul places
one weight shard per DRAM bank and runs a dedicated compute core per bank-bucket,
which removes the 130-core bank contention of the interleaved kernel (the root
cause identified in the plan's tracy).
"""
import math

import ttnn

TILE = ttnn.TILE_SIZE  # 32


def find_grid_k_n(K_tiles: int, N_tiles: int, max_rows: int = 8, max_cols: int = 8):
    """Largest num_cores (<= max_rows*max_cols) dividing BOTH K_tiles and N_tiles.

    Returns (rows, cols). Mirrors tt_transformers ``find_grid_k_n``.
    """
    max_cores = max_rows * max_cols
    possible = [c for c in range(1, max_cores + 1) if K_tiles % c == 0 and N_tiles % c == 0]
    possible.sort(reverse=True)
    for cores in possible:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols
    raise AssertionError(f"no grid for K_tiles={K_tiles} N_tiles={N_tiles}")


def num_cores_for_k_n(k: int, n: int) -> int:
    rows, cols = find_grid_k_n(k // TILE, n // TILE)
    return rows * cols


def find_largest_divisor(n: int, max_divisor: int = 8) -> int:
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


def create_dram_sharded_mem_config(device, k: int, n: int) -> ttnn.MemoryConfig:
    """WIDTH_SHARDED DRAM weight memory config across the device's DRAM banks.

    k = in-features (shard height), n = out-features (sharded over banks).
    """
    dram_cores = device.dram_grid_size().x  # 8 on p150
    padded_n = math.ceil(n / (TILE * dram_cores)) * (TILE * dram_cores)
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))})
    shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_n // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def dram_matmul_config(m: int, k: int, n: int, num_cores: int, fused_activation=None):
    """MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig for a decode matmul.

    Strict: k % (TILE * num_cores) == 0 (validated by the caller).
    """
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=find_largest_divisor(k // (TILE * num_cores)),
        per_core_M=math.ceil(m / TILE),
        per_core_N=math.ceil(n / (TILE * num_cores)),
        fused_activation=fused_activation,
    )


def sharded_matmul_plan(device, m: int, k: int, n: int):
    """Return (num_cores, program_config) for a DRAM-sharded decode matmul, or
    (None, None) if the shape is not expressible as a DRAM-sharded matmul.

    Validates the strictness rules so a bad shape falls back to interleaved at
    the call site instead of TT_FATAL'ing.
    """
    if k % TILE != 0 or n % TILE != 0:
        return None, None
    try:
        num_cores = num_cores_for_k_n(k, n)
    except AssertionError:
        return None, None
    if num_cores == 0 or k % (TILE * num_cores) != 0:
        return None, None
    pc = dram_matmul_config(m, k, n, num_cores)
    return num_cores, pc


def core_grid_for_num_cores(num_cores: int, max_cols: int = 8) -> ttnn.CoreGrid:
    """Pick a (rows, cols) CoreGrid with rows*cols == num_cores (cols <= max_cols)."""
    for cols in range(min(max_cols, num_cores), 0, -1):
        if num_cores % cols == 0:
            return ttnn.CoreGrid(x=cols, y=num_cores // cols)
    return ttnn.CoreGrid(x=1, y=num_cores)


def width_sharded_l1_config(m: int, width: int, num_cores: int) -> ttnn.MemoryConfig:
    """L1 WIDTH_SHARDED memory config for a [m, width] activation over num_cores.

    Shard shape (m, width/num_cores). width must be divisible by num_cores.
    """
    grid = core_grid_for_num_cores(num_cores)
    return ttnn.create_sharded_memory_config(
        shape=(m, width // num_cores),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def create_sharded_norm_config(grid: ttnn.CoreGrid, dim: int, block_h_tiles: int):
    """LayerNormShardedMultiCoreProgramConfig for a width-sharded RMSNorm.

    grid: the CoreGrid the input activation is width-sharded across.
    dim: feature width (1536).
    block_h_tiles: rows / TILE (1 for a single decode tile of 32 rows).
    """
    block_w = dim // grid.num_cores // TILE
    subblock_w = 4
    while subblock_w > 0:
        if block_w % subblock_w == 0:
            break
        subblock_w -= 1
    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[grid.x, grid.y],
        subblock_w=subblock_w,
        block_h=block_h_tiles,
        block_w=block_w,
        inplace=False,
    )

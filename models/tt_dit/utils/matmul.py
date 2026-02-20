# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

from loguru import logger

import ttnn

# Track unique warning signatures to avoid stdout spam
_warned_matmul_signatures = set()


# Known best blockings for 8x8 core grid for specific (M, K, N) shapes
# Each value is a tuple: (M_block_size, K_block_size, N_block_size)
grid_88_configs = {
    (32, 2432, 7296): (2, 4, 8),
    (1024, 2432, 3648): (4, 4, 16),
    (352, 2432, 3648): (2, 4, 8),
    (1024, 2432, 1216): (4, 4, 8),
    (352, 2432, 1216): (2, 4, 8),
    (1024, 2432, 4864): (4, 4, 8),
    (1024, 4864, 2432): (4, 4, 16),
    (352, 2432, 4864): (4, 8, 4),
    (352, 4864, 2432): (2, 8, 4),
    (32, 3072, 6144): (2, 8, 8),
    (32, 3072, 3072): (2, 4, 16),
    (11264, 3072, 4608): (16, 8, 4),
    (128, 1536, 4608): (2, 8, 4),
    (11264, 3072, 1536): (8, 8, 8),
    (128, 3072, 768): (2, 8, 4),
    (11264, 3072, 8192): (16, 8, 4),
    (11264, 4096, 3072): (16, 4, 4),
    (128, 1536, 4096): (2, 8, 4),
    (128, 2048, 1536): (2, 4, 8),
    (18944, 5120, 2560): (8, 8, 8),
    (128, 5120, 2560): (4, 16, 2),
    (18944, 5120, 6912): (16, 8, 4),
    (18944, 6912, 5120): (8, 8, 8),
    (32, 2432, 3648): (2, 4, 8),
    (1024, 2432, 1920): (4, 4, 8),
    (352, 2432, 1920): (2, 4, 8),
    (1024, 2560, 608): (4, 4, 4),
    (352, 2560, 608): (2, 5, 4),
    (1024, 2432, 2432): (4, 4, 16),
    (352, 2432, 2432): (2, 4, 16),
    (32, 3072, 1536): (2, 4, 8),
    (5632, 3072, 2304): (8, 12, 4),
    (128, 1536, 2304): (2, 6, 4),
    (5632, 3072, 768): (8, 8, 4),
    (128, 3072, 384): (2, 6, 2),
    (5632, 3072, 4096): (8, 6, 8),
    (5632, 2048, 3072): (8, 8, 8),
    (128, 1536, 2048): (2, 6, 4),
    (128, 1024, 1536): (2, 4, 4),
    (9472, 5120, 1280): (8, 10, 8),
    (128, 5120, 1280): (2, 8, 8),
    (9472, 5120, 3456): (8, 10, 8),
    (9472, 3456, 5120): (8, 12, 4),
}


# Known best blockings for 8x9 core grid for specific (M, K, N) shapes
# Each value is a tuple: (M_block_size, K_block_size, N_block_size)
grid_89_configs = {
    (32, 2432, 3648): (2, 4, 8),
    (1024, 2432, 1920): (4, 4, 8),
    (352, 2432, 1920): (2, 4, 4),
    (1024, 2560, 608): (4, 4, 4),
    (352, 2560, 608): (2, 4, 4),
    (1024, 2432, 2432): (4, 4, 16),
    (352, 2432, 2432): (2, 4, 16),
    (32, 3072, 3072): (2, 4, 16),
    (32, 3072, 1536): (2, 4, 8),
    (5632, 3072, 2304): (8, 8, 8),
    (128, 1536, 2304): (2, 8, 2),
    (5632, 3072, 768): (8, 8, 4),
    (128, 3072, 384): (2, 4, 4),
    (5632, 3072, 4096): (8, 8, 8),
    (5632, 2048, 3072): (8, 8, 4),
    (128, 1536, 2048): (4, 4, 4),
    (128, 1024, 1536): (2, 4, 4),
    (9472, 5120, 1280): (8, 8, 8),
    (128, 5120, 1280): (2, 16, 2),
    (9472, 5120, 3456): (8, 8, 8),
    (9472, 3456, 5120): (8, 8, 8),
}

grid_13_9_configs = {
    (9472, 5120, 1280): (8, 8, 8, (2, 2)),
    (128, 5120, 1280): (2, 16, 4, (2, 2)),
    (9472, 5120, 3456): (8, 8, 4, (1, 2)),
    (9472, 3456, 5120): (8, 12, 4, (1, 2)),
}

grid_12_10_configs = {
    (9472, 5120, 1280): (16, 8, 4, (2, 2)),
    (128, 5120, 1280): (1, 16, 8, (1, 2)),
    (9472, 5120, 3456): (16, 8, 4, (1, 2)),
    (9472, 3456, 5120): (8, 4, 8, (1, 2)),
}


def get_matmul_config(M, K, N, core_grid):
    # Default to 8x8x8 with subblock 2x2 when unknown
    subblock_h = 2
    subblock_w = 2

    # Fallback core grid if not provided
    if core_grid is None:
        core_grid = ttnn.CoreCoord(8, 8)

    config_tuple = None
    # Only use lookup for 8x8 grid; otherwise default
    if getattr(core_grid, "x", None) == 8 and getattr(core_grid, "y", None) == 8:
        config_tuple = grid_88_configs.get((M, K, N))
    elif getattr(core_grid, "x", None) == 8 and getattr(core_grid, "y", None) == 9:
        config_tuple = grid_89_configs.get((M, K, N))
    elif getattr(core_grid, "x", None) == 13 and getattr(core_grid, "y", None) == 9:
        config_tuple = grid_13_9_configs.get((M, K, N))
        if config_tuple is not None:
            subblock_h, subblock_w = config_tuple[3]
            config_tuple = config_tuple[:3]
    elif getattr(core_grid, "x", None) == 12 and getattr(core_grid, "y", None) == 10:
        config_tuple = grid_12_10_configs.get((M, K, N))
        if config_tuple is not None:
            subblock_h, subblock_w = config_tuple[3]
            config_tuple = config_tuple[:3]

    if config_tuple is None:
        M_block_size, K_block_size, N_block_size = 8, 8, 8

        M_tiles = math.ceil(M / 32)
        N_tiles = math.ceil(N / 32)

        if M_tiles < M_block_size:
            M_block_size = subblock_h
        if N_tiles < N_block_size:
            N_block_size = subblock_w

        grid_x = getattr(core_grid, "x", None)
        grid_y = getattr(core_grid, "y", None)
        signature = (M, K, N, grid_x, grid_y)
        if signature not in _warned_matmul_signatures:
            logger.warning(
                f"No known best blocking for (M, K, N) = ({M}, {K}, {N}) on {grid_x}x{grid_y} core grid; using default {M_block_size}x{K_block_size}x{N_block_size}"
            )
            _warned_matmul_signatures.add(signature)
    else:
        M_block_size, K_block_size, N_block_size = config_tuple

    return ttnn.MinimalMatmulConfig(
        M_block_size=M_block_size,
        K_block_size=K_block_size,
        N_block_size=N_block_size,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        compute_with_storage_grid_size=core_grid,
    )

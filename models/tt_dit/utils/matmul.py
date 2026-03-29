# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
from typing import NamedTuple

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
    (9472, 5120, 3840): (16, 8, 4, (1, 2)),
    (9472, 3456, 5120): (8, 4, 8, (1, 2)),
}

grid_12_9_configs = {
    (9472, 5120, 1280): (10, 8, 8, (2, 1)),
    (2368, 5120, 1280): (10, 8, 6, (2, 1)),
    (128, 5120, 1280): (1, 16, 8, (1, 2)),
    (9472, 5120, 3456): (9, 5, 12, (1, 2)),
    (2368, 5120, 3456): (7, 5, 12, (1, 2)),
    (9472, 5120, 3840): (7, 5, 16, (1, 2)),
    (2368, 5120, 3840): (7, 5, 16, (1, 2)),
    # (6144, 5120, 1280): (8, 20, 2, (4, 1)),
    # (6144, 5120, 3456): (4, 40, 3, (1, 3)),
    (6144, 5120, 3840): (6, 5, 16, (1, 1)),
    # (6240, 5120, 1280): (10, 16, 4, (1, 4)),
    # (6240, 5120, 3840): (5, 40, 1, (1, 1)),
    # (14400, 5120, 1280): (10, 10, 8, (1, 1)),
    # (14400, 5120, 3840): (6, 20, 3, (1, 3)),
}

grid_11_10_configs = {
    (512, 5120, 2560): (2, 5, 8, (1, 4)),
    (2368, 5120, 3840): (4, 8, 12, (1, 4)),
    (2368, 5120, 1280): (8, 4, 4, (1, 4)),
    (2368, 5120, 3456): (8, 2, 12, (2, 2)),
    (2368, 3456, 5120): (4, 4, 8, (1, 4)),
    (9472, 5120, 3840): (16, 4, 4, (1, 4)),
    (9472, 5120, 1280): (16, 8, 4, (1, 4)),
    (9472, 5120, 3456): (16, 8, 4, (1, 4)),
    (9472, 3456, 5120): (16, 3, 4, (1, 4)),
    (32, 32, 32): (1, 1, 1, (1, 1)),
    (32, 96, 192): (1, 2, 4, (1, 4)),
    (32, 192, 384): (1, 6, 12, (1, 4)),
    (32, 256, 5120): (1, 6, 10, (1, 1)),
    (32, 1280, 30720): (1, 20, 5, (1, 1)),
    (32, 3072, 10240): (1, 24, 2, (1, 2)),
    (32, 5120, 1280): (1, 32, 1, (1, 1)),
    (32, 10240, 10240): (1, 64, 3, (1, 3)),
    (64, 96, 192): (1, 2, 4, (1, 1)),
    (64, 192, 384): (1, 5, 5, (1, 1)),
    (96, 96, 192): (1, 3, 3, (1, 1)),
    (128, 5120, 2560): (1, 12, 20, (1, 2)),
    (512, 4096, 5120): (2, 32, 6, (2, 1)),
    (512, 5120, 5120): (2, 32, 5, (2, 1)),
    (6144, 384, 384): (10, 6, 4, (2, 2)),
    (6144, 384, 1152): (6, 12, 3, (1, 3)),
    (6144, 3456, 5120): (6, 12, 10, (1, 2)),
    (6144, 5120, 64): (12, 2, 2, (3, 1)),
    (6144, 5120, 3456): (6, 12, 8, (1, 1)),
    (6240, 384, 384): (20, 8, 1, (4, 1)),
    (6240, 384, 1152): (6, 12, 3, (1, 3)),
    (6240, 3456, 5120): (6, 20, 3, (1, 1)),
    (6240, 5120, 64): (10, 3, 1, (2, 1)),
    (6240, 5120, 3456): (6, 20, 5, (2, 1)),
    (14400, 384, 384): (9, 12, 3, (3, 1)),
    (14400, 384, 1152): (6, 12, 2, (1, 1)),
    (14400, 3456, 5120): (15, 12, 4, (1, 2)),
    (14400, 5120, 3456): (15, 20, 1, (3, 1)),
    # (14400, 5120, 64): (15, 10, 1, (3, 1)),
}


_BH_GALAXY_MIN_DEVICES = 32
_BH_GALAXY_MAX_CORE_GRID = (11, 10)


def get_matmul_core_grid(mesh_device):
    """Return the compute core grid, clamped to 11x10 on Blackhole Galaxy (power constraint)."""
    core_grid = mesh_device.compute_with_storage_grid_size()
    if mesh_device.get_num_devices() >= _BH_GALAXY_MIN_DEVICES:
        core_grid = ttnn.CoreCoord(
            min(core_grid.x, _BH_GALAXY_MAX_CORE_GRID[0]),
            min(core_grid.y, _BH_GALAXY_MAX_CORE_GRID[1]),
        )
    return core_grid


def get_matmul_config(M, K, N, core_grid, default_block_size=None):
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
    elif getattr(core_grid, "x", None) == 11 and getattr(core_grid, "y", None) == 10:
        config_tuple = grid_11_10_configs.get((M, K, N))
        if config_tuple is not None:
            subblock_h, subblock_w = config_tuple[3]
            config_tuple = config_tuple[:3]
    elif getattr(core_grid, "x", None) == 12 and getattr(core_grid, "y", None) == 9:
        config_tuple = grid_12_9_configs.get((M, K, N))
        if config_tuple is not None:
            subblock_h, subblock_w = config_tuple[3]
            config_tuple = config_tuple[:3]

    if config_tuple is None:
        M_block_size, K_block_size, N_block_size = default_block_size if default_block_size is not None else (8, 8, 8)

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


class FusedMMRSConfig(NamedTuple):
    compute_with_storage_grid_size: ttnn.CoreCoord
    M_block_size: int
    K_block_size: int
    N_block_size: int
    subblock_h: int
    subblock_w: int
    num_buffers_per_channel: int | None
    chunk_width_in_mm_blocks: int

    def get_params(self, core_grid, num_links):
        rs_zone_capacity = (core_grid.y - self.compute_with_storage_grid_size.y) * core_grid.x
        num_workers_per_link = rs_zone_capacity // (2 * num_links) - 1
        config_dict = self._asdict()
        num_buffers_per_channel = config_dict.pop("num_buffers_per_channel")
        chunk_width_in_mm_blocks = config_dict.pop("chunk_width_in_mm_blocks")

        # Order is important. Guaranteed for python 3.7+
        return {
            "reduce_scatter_core_grid_offset": ttnn.CoreCoord(0, self.compute_with_storage_grid_size.y),
            "num_links": num_links,
            "config": ttnn.MinimalMatmulConfig(**config_dict),
            "num_buffers_per_channel": num_buffers_per_channel,
            "chunk_width_in_mm_blocks": chunk_width_in_mm_blocks,
            "num_workers_per_link": num_workers_per_link,
        }


default_fused_mmrs_config = FusedMMRSConfig(ttnn.CoreCoord(8, 7), 2, 8, 8, 1, 1, None, 1)
# core_grid: {MKN: mm_core_grid, M, K, N, sub_h, sub_w, num_w_p_link, num_buffers_per_channel, chunk_width_in_mm_blocks}
fused_mmrs_configs = {
    ttnn.CoreCoord(8, 9): {
        (9472, 5120, 1280): FusedMMRSConfig(ttnn.CoreCoord(8, 7), 8, 8, 8, 2, 2, None, 1),
    },
    ttnn.CoreCoord(12, 10): {
        (9472, 3456, 5120): FusedMMRSConfig(ttnn.CoreCoord(12, 8), 8, 4, 8, 2, 1, None, 1),
        (9472 // 4, 3456, 5120): FusedMMRSConfig(ttnn.CoreCoord(12, 8), 4, 4, 8, 2, 2, None, 1),
    },
}


def get_fused_mmrs_config(M, K, N, device_core_grid, num_links):
    config = fused_mmrs_configs.get(device_core_grid, {})
    if len(config) == 0:
        logger.warning(
            f"No known best MM/RS blocking for (M, K, N) = ({M}, {K}, {N}) on {device_core_grid} core grid; using default"
        )
    config = config.get((M, K, N), default_fused_mmrs_config)
    return config.get_params(device_core_grid, num_links)

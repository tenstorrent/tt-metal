# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

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
    (9472, 5120, 3456): (10, 8, 8),
    (9472, 5120, 3840): (10, 8, 8),
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

grid_11_10_configs = {
    (512, 5120, 2560): (2, 5, 8, (1, 4)),
    (512, 4096, 1024): (2, 8, 4, (1, 4)),
    (512, 4096, 2560): (2, 4, 8, (1, 4)),
    (512, 2560, 4096): (2, 4, 16, (1, 4)),
    (512, 4096, 5120): (2, 8, 8, (1, 4)),
    (512, 5120, 5120): (2, 5, 8, (1, 4)),
    (32, 256, 5120): (1, 2, 8, (1, 4)),
    (32, 5120, 1280): (1, 10, 4, (1, 4)),
    (32, 1280, 30720): (1, 20, 4, (1, 4)),
    (32, 192, 384): (1, 3, 2, (1, 2)),
    (64, 192, 384): (1, 3, 2, (1, 2)),
    (2368, 5120, 64): (4, 8, 2, (2, 2)),
    (2368, 5120, 3840): (4, 8, 12, (1, 4)),
    (2368, 5120, 1280): (8, 4, 4, (1, 4)),
    (2368, 5120, 3456): (8, 2, 12, (2, 2)),
    (2368, 3456, 5120): (4, 4, 8, (1, 4)),
    (9472, 5120, 64): (2, 32, 2, (2, 2)),
    (9472, 5120, 3840): (16, 4, 4, (1, 4)),
    (9472, 5120, 1280): (16, 8, 4, (1, 4)),
    (9472, 5120, 3456): (16, 8, 4, (1, 4)),
    (9472, 3456, 5120): (16, 3, 4, (1, 4)),
    (14400, 384, 1152): (16, 3, 8, (2, 2)),
    (14400, 384, 384): (8, 3, 4, (2, 2)),
}


_BH_GALAXY_MIN_DEVICES = 32
_BH_GALAXY_MAX_CORE_GRID = (11, 10)


def get_matmul_core_grid(mesh_device):
    """Return the compute core grid, clamped to 11x10 on Blackhole Galaxy (power constraint)."""
    core_grid = mesh_device.compute_with_storage_grid_size()
    if ttnn.device.is_blackhole() and mesh_device.get_num_devices() >= _BH_GALAXY_MIN_DEVICES:
        core_grid = ttnn.CoreCoord(
            min(core_grid.x, _BH_GALAXY_MAX_CORE_GRID[0]),
            min(core_grid.y, _BH_GALAXY_MAX_CORE_GRID[1]),
        )
    return core_grid


grid_12_9_configs = {
    (9472, 5120, 1280): (10, 8, 8, (2, 1)),
    (2368, 5120, 1280): (10, 8, 6, (2, 1)),
    (128, 5120, 1280): (1, 16, 8, (1, 2)),
    (9472, 5120, 3456): (9, 5, 12, (1, 2)),
    (2368, 5120, 3456): (7, 5, 12, (1, 2)),
    (9472, 5120, 3840): (7, 5, 16, (1, 2)),
    (2368, 5120, 3840): (7, 5, 16, (1, 2)),
}


def get_matmul_config(M, K, N, core_grid, default_block_size=None):
    # Default to 8x8x8 with subblock 2x2 when unknown
    subblock_h = 2
    subblock_w = 2

    # Fallback core grid if not provided
    if core_grid is None:
        core_grid = ttnn.CoreCoord(8, 8)

    config_tuple = None
    grid_x = getattr(core_grid, "x", None)
    grid_y = getattr(core_grid, "y", None)
    grid_lookup = {
        (8, 8): grid_88_configs,
        (8, 9): grid_89_configs,
        (13, 9): grid_13_9_configs,
        (12, 10): grid_12_10_configs,
        (11, 10): grid_11_10_configs,
        (12, 9): grid_12_9_configs,
    }
    grid_dict = grid_lookup.get((grid_x, grid_y))
    if grid_dict is not None:
        config_tuple = grid_dict.get((M, K, N))

    # Unpack: 3-tuple (M_block_size, K_block_size, N_block_size) or
    # 4-tuple (M_block_size, K_block_size, N_block_size, (sub_h, sub_w))
    if config_tuple is not None and len(config_tuple) == 4:
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


def register_matmul_configs(configs: dict) -> None:
    """Register additional matmul block-size configs from external models.

    Args:
        configs: Mapping from grid key string to dict of (M,K,N) -> config tuples.
            Grid keys: ``"11x10"``, ``"12x10"``, ``"12x9"``, ``"13x9"``, ``"8x8"``, ``"8x9"``.
            Config tuple format: ``(M_block, K_block, N_block)`` or
            ``(M_block, K_block, N_block, (sub_h, sub_w))``.
            When subblock is omitted, the default ``(2, 2)`` is used.

    Example::

        register_matmul_configs({
            "11x10": {
                (14400, 384, 384): (9, 12, 3, (3, 1)),
                (14400, 5120, 3456): (15, 20, 1, (3, 1)),
            },
        })
    """
    grid_map = {
        "8x8": grid_88_configs,
        "8x9": grid_89_configs,
        "11x10": grid_11_10_configs,
        "12x10": grid_12_10_configs,
        "12x9": grid_12_9_configs,
        "13x9": grid_13_9_configs,
    }
    for grid_key, entries in configs.items():
        target = grid_map.get(grid_key)
        if target is None:
            msg = f"Unknown grid key {grid_key!r}, expected one of {list(grid_map)}"
            raise ValueError(msg)
        target.update(entries)


def register_fused_mmrs_configs(configs: dict) -> None:
    """Register additional fused matmul+reduce-scatter configs.

    Args:
        configs: Mapping from ``ttnn.CoreCoord`` to dict of
            ``(M,K,N)`` -> :class:`FusedMMRSConfig`.

    Example::

        register_fused_mmrs_configs({
            ttnn.CoreCoord(12, 10): {
                (14400, 3456, 5120): FusedMMRSConfig(
                    ttnn.CoreCoord(12, 8), 8, 4, 8, 2, 1, None, 1
                ),
            },
        })
    """
    for core_grid, entries in configs.items():
        fused_mmrs_configs.setdefault(core_grid, {}).update(entries)

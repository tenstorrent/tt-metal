# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import os
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
    # LTX Gemma encoder + connectors (seq=1024, swept on 110 cores; M_block=4 beats the 8x8x8
    # default by ~10-13% on the FFN/proj shapes; the FE aggregate is compute-bound, ~flat).
    (1024, 3840, 3840): (4, 8, 8, (1, 4)),
    (1024, 4096, 4096): (4, 8, 12, (1, 4)),
    (1024, 3840, 2048): (4, 8, 8, (1, 4)),
    (1024, 4096, 960): (4, 8, 4, (1, 4)),
    (1024, 15360, 960): (4, 8, 4, (1, 4)),
    (1024, 4096, 1024): (4, 8, 12, (1, 4)),
    (1024, 188160, 4096): (4, 8, 12, (1, 4)),
    # FE aggregate after column-parallel sharding on TP=4: video 4096/4=1024, audio 2048/4=512.
    (1024, 188160, 1024): (4, 8, 8, (1, 4)),
    (1024, 188160, 512): (4, 8, 4, (1, 4)),

    (32, 4096, 1024): (2, 8, 4, (2, 2)),
    (32, 4096, 2560): (2, 4, 12, (2, 2)),
    (32, 2560, 4096): (2, 4, 12, (2, 2)),
    (320, 512, 1024): (2, 4, 4, (2, 2)),
    (320, 1024, 256): (2, 4, 2, (2, 2)),
    (320, 1024, 1024): (2, 4, 4, (2, 2)),
    (3840, 384, 1152): (11, 4, 4, (1, 4)),
    (3840, 384, 384): (11, 4, 2, (1, 2)),
    (15360, 384, 1152): (22, 4, 4, (2, 2)),
    (15360, 384, 384): (11, 3, 4, (1, 4)),
    (32, 96, 192): (2, 3, 2, (2, 2)),
    (64, 96, 192): (2, 3, 2, (2, 2)),
    (32, 32, 32): (2, 1, 2, (2, 2)),
    (96, 8192, 5120): (6, 16, 8, (2, 2)),
    (96, 5120, 5120): (4, 8, 8, (2, 2)),
    (96, 5120, 2560): (6, 5, 8, (2, 2)),
    (96, 2048, 5120): (2, 4, 8, (2, 2)),
    (2656, 5120, 64): (4, 8, 2, (2, 2)),
    (32, 256, 512): (2, 4, 2, (2, 2)),
    (32, 512, 128): (2, 8, 2, (2, 2)),
    (32, 128, 30720): (2, 2, 8, (2, 2)),
    (11520, 5120, 64): (3, 40, 2, (3, 1)),
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
    (1216, 4096, 32): (8, 8, 1, (4, 1)),
    (1216, 4096, 3072): (4, 8, 12, (1, 4)),
    (1216, 4096, 1024): (4, 8, 4, (1, 4)),
    (1216, 4096, 512): (4, 8, 2, (2, 2)),
    (1216, 2048, 1024): (4, 8, 4, (1, 4)),
    (1216, 4096, 4096): (4, 8, 16, (1, 4)),
    (4864, 4096, 32): (20, 8, 1, (4, 1)),
    (4864, 4096, 3072): (10, 4, 12, (1, 4)),
    (4864, 4096, 1024): (16, 8, 4, (4, 1)),
    (4864, 4096, 512): (8, 8, 2, (4, 1)),
    (4864, 2048, 1024): (5, 8, 4, (1, 4)),
    (4864, 4096, 4096): (5, 8, 16, (1, 4)),
    (256, 2048, 1024): (2, 8, 4, (1, 4)),
    (32, 2048, 32): (1, 8, 1, (1, 1)),
    (32, 2048, 1536): (1, 4, 16, (1, 4)),
    (32, 2048, 512): (1, 8, 2, (1, 2)),
    (32, 2048, 2048): (1, 4, 12, (1, 4)),
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
    # Optional explicit reduce-scatter worker count
    num_workers_per_link: int | None = None

    def get_params(self, core_grid, num_links):
        config_dict = self._asdict()
        num_buffers_per_channel = config_dict.pop("num_buffers_per_channel")
        chunk_width_in_mm_blocks = config_dict.pop("chunk_width_in_mm_blocks")
        num_workers_override = config_dict.pop("num_workers_per_link")

        if num_workers_override is not None:
            num_workers_per_link = num_workers_override
        else:
            rs_zone_capacity = (core_grid.y - self.compute_with_storage_grid_size.y) * core_grid.x
            num_workers_per_link = rs_zone_capacity // (2 * num_links) - 1

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
        # LTX video FFN ff2 (RowParallel): per-device [4864,4096]@[4096,4096]
        (4864, 4096, 4096): FusedMMRSConfig(ttnn.CoreCoord(12, 8), 7, 5, 6, 1, 3, None, 1, 3),
    },
}


def get_fused_mmrs_config(M, K, N, device_core_grid, num_links):
    config = fused_mmrs_configs.get(device_core_grid, {})
    if len(config) == 0:
        logger.warning(
            f"No known best MM/RS blocking for (M, K, N) = ({M}, {K}, {N}) on {device_core_grid} core grid; using default"
        )
    elif (M, K, N) not in config:
        # Worth a warning even though the grid is known: the default puts the matmul on an 8x7 grid at
        # subblock 1x1, so on a larger device it is drastically slower than the unfused
        # matmul + reduce-scatter it is meant to replace, and the fusion reads as a regression.
        logger.warning(
            f"No known best MM/RS blocking for (M, K, N) = ({M}, {K}, {N}) on {device_core_grid} core grid; "
            "using default, which is likely slower than not fusing at all"
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


# ===================================================================== Fabric-bound all-gather-matmul
class FabricAGMMConfig(NamedTuple):
    mm_core_grid: ttnn.CoreCoord
    ag_core_grid_offset: tuple
    M_block_size: int
    K_block_size: int
    N_block_size: int
    subblock_h: int
    subblock_w: int
    num_workers_per_link: int
    num_buffers_per_channel: int


# Keyed by device core-grid (``ttnn.CoreCoord``) then ``(K, N, chunks)``
fabric_agmm_configs: dict[ttnn.CoreCoord, dict[tuple, FabricAGMMConfig]] = {
    ttnn.CoreCoord(12, 10): {
        # (K, N, chunks) -> FabricAGMMConfig
        (4096, 1024, 1): FabricAGMMConfig(ttnn.CoreCoord(12, 8), (0, 8), 16, 8, 4, 2, 2, 3, 8),
        # --------------------------------------------------------------------------------------- Remaining
        (4096, 32, 1): FabricAGMMConfig(ttnn.CoreCoord(12, 8), (0, 8), 16, 8, 1, 2, 1, 3, 8),
        # audio to_gate_logits: per-device N = num_heads/tp = 8, tile-padded to 32 (1 tile)
        (4096, 512, 1): FabricAGMMConfig(ttnn.CoreCoord(12, 8), (0, 8), 16, 8, 2, 2, 2, 3, 8),
        # audio a_kv (chunks=1 in the op test): K = audio_dim = 2048, N = 1024
    },
}


def get_fabric_agmm_config(K, N, chunks, device_core_grid) -> FabricAGMMConfig | None:
    """Return the tuned fabric-bound strided-AGMM config for this shape, or ``None``.

    ``None`` means the shape is not (known to be) fabric-bound; the caller keeps the current
    ``all_gather_minimal_matmul_async`` path. Keyed on ``(K, N, chunks)`` only (M-independent).

    A/B switch: set ``DISABLE_FABRIC_AGMM=1`` to force a miss for every shape, routing the whole
    model back onto the old ``all_gather_minimal_matmul_async`` op. Used to get an apples-to-apples
    old-agmm-vs-strided-sagmm baseline under the same trace/fabric config.
    """
    if os.environ.get("DISABLE_FABRIC_AGMM") in ("1", "true", "True"):
        return None
    return fabric_agmm_configs.get(device_core_grid, {}).get((K, N, chunks))


def register_fabric_agmm_configs(configs: dict) -> None:
    """Register additional fabric-bound strided-AGMM configs from external models.

    Args:
        configs: Mapping from ``ttnn.CoreCoord`` (device core-grid) to dict of
            ``(K, N, chunks)`` -> :class:`FabricAGMMConfig`.

    Example::

        register_fabric_agmm_configs({
            ttnn.CoreCoord(12, 10): {
                (4096, 1024, 1): FabricAGMMConfig(
                    ttnn.CoreCoord(12, 8), (0, 8), 16, 8, 4, 2, 2, 3, 8
                ),
            },
        })
    """
    for core_grid, entries in configs.items():
        fabric_agmm_configs.setdefault(core_grid, {}).update(entries)

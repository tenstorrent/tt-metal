# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The device tile blend and unpatchify against their host originals, at the production geometry.

Kept separate from ``test_vae_parallel_minimax_h3.py`` because these run ``FABRIC_1D_RING``
while that file runs ``FABRIC_1D``, and ``fabric_config`` is a process-global one-shot: a second
distinct value in one process raises ``TT_FATAL: Tried to override previous value of fabric config``.

This is the piece that has to be right before any of it is worth doing. The reference blend is
sequential and asymmetric; a separable reformulation moves 11.1 % of pixels by up to 4.66 (measured),
so the device version mirrors the order rather than the algebra. These tests are what say it does.

`single_device` throughout: the question here is the arithmetic, not the distribution. The
all-gather that co-locates neighbouring tiles is a separate concern and gated separately.
"""

import pytest
import torch
from loguru import logger

import ttnn

from ....models.vae.minimax_h3.decoder_minimax_h3 import unpatchify
from ....models.vae.minimax_h3.stitch_device_minimax_h3 import DeviceTileStitcher, unpatchify_device
from ....models.vae.minimax_h3.vae_minimax_h3 import MiniMaxH3VaeConfig, split_tiles, stitch_tiles
from ....utils.check import assert_quality

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]

# 1344x768 with the real tile size and overlap: a 4x7 grid, overlaps [96, 80, 80] by height and
# [80, 80, 80, 80, 64, 64] by width. Derived rather than hardcoded -- a hardcoded overlap of 32
# puts the seam bands on non-boundary columns and misses the actual seams.
HEIGHT, WIDTH = 768, 1344
# One decoder work unit covers 7 latent frames -> 28 pixel frames. Trimmed to 4 for these tests: the
# blend is per-pixel along H and W and does nothing along T, so T is a multiplier on cost only.
PIXEL_FRAMES = 4
CHANNELS = 3


def _geometry():
    config = MiniMaxH3VaeConfig()
    ratio = config.spatial_compression_ratio
    _, _, height_overlaps = split_tiles(HEIGHT, 256, 64, ratio)
    _, _, width_overlaps = split_tiles(WIDTH, 256, 64, ratio)
    return height_overlaps, width_overlaps


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_stitch_matches_host_at_production_geometry(mesh_device, reset_seeds):
    """The whole 4x7 stitch, device against host.

    The bar is `pcc=0.9999` with `relative_rmse` paired: the blend output is consumed as an absolute
    pixel value, and a seam is a *local* defect that a whole-canvas PCC can dilute -- so the seam
    columns are also checked on their own below.

    Single-blend coverage is included: the stitch runs every cross-fade the stitcher has, on both
    axes at every real overlap extent (96/80 by height, 80/64 by width), and the per-seam bands
    below hold each one to the same `pcc=0.9999` bar against the host original.
    """
    height_overlaps, width_overlaps = _geometry()
    rows, columns = len(height_overlaps) + 1, len(width_overlaps) + 1
    logger.info(f"grid {rows}x{columns} = {rows * columns} tiles, overlaps h={height_overlaps} w={width_overlaps}")

    tiles = [[torch.randn(1, CHANNELS, PIXEL_FRAMES, 256, 256) for _ in range(columns)] for _ in range(rows)]
    expected = stitch_tiles(tiles, height_overlaps, width_overlaps)
    assert expected.shape[-2:] == (HEIGHT, WIDTH), f"host stitch gave {tuple(expected.shape[-2:])}"

    stitcher = DeviceTileStitcher(mesh_device)
    device_tiles = [
        [ttnn.from_torch(t, dtype=ttnn.float32, device=mesh_device, layout=ttnn.TILE_LAYOUT) for t in row]
        for row in tiles
    ]
    actual = ttnn.to_torch(stitcher.stitch(device_tiles, height_overlaps, width_overlaps))

    assert actual.shape == expected.shape, f"{tuple(actual.shape)} != {tuple(expected.shape)}"
    assert_quality(expected, actual, pcc=0.9999, relative_rmse=0.02)

    # The seams specifically. A whole-canvas metric averages a boundary defect away, which is the
    # failure mode this whole exercise is trying not to introduce.
    y = 0
    for index, overlap in enumerate(height_overlaps):
        y += 256 - overlap
        band_expected, band_actual = expected[..., y : y + overlap, :], actual[..., y : y + overlap, :]
        logger.info(f"horizontal seam {index} at y={y}, extent {overlap}")
        assert_quality(band_expected, band_actual, pcc=0.9999)
    x = 0
    for index, overlap in enumerate(width_overlaps):
        x += 256 - overlap
        logger.info(f"vertical seam {index} at x={x}, extent {overlap}")
        assert_quality(expected[..., x : x + overlap], actual[..., x : x + overlap], pcc=0.9999)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_unpatchify_matches_host(mesh_device, reset_seeds):
    """The 8-dimensional permute on device against the torch original."""
    config = MiniMaxH3VaeConfig()
    num_frames, height, width = 7, 16, 16
    patch, patch_t = config.spatial_compression_ratio, config.temporal_compression_ratio
    tokens = torch.randn(1, num_frames * height * width, CHANNELS * patch_t * patch * patch)

    expected = unpatchify(
        tokens,
        num_frames=num_frames,
        height=height,
        width=width,
        out_channels=CHANNELS,
        patch_size=patch,
        patch_size_t=patch_t,
    )
    device_tokens = ttnn.from_torch(tokens, dtype=ttnn.float32, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    actual = ttnn.to_torch(
        unpatchify_device(
            device_tokens,
            num_frames=num_frames,
            height=height,
            width=width,
            out_channels=CHANNELS,
            patch_size=patch,
            patch_size_t=patch_t,
        )
    )

    logger.info(f"unpatchify {tuple(tokens.shape)} -> {tuple(expected.shape)}")
    assert actual.shape == expected.shape, f"{tuple(actual.shape)} != {tuple(expected.shape)}"
    assert_quality(expected, actual, pcc=0.9999)


# ---------------------------------------------------------------------------
# The two-axis all-gather's batch permutation
# ---------------------------------------------------------------------------

MESH_4X8 = [
    pytest.param(
        (4, 8),
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "require_exact_physical_num_devices": True},
        id="4x8",
    )
]


def gathered_tile_order(mesh_rows: int, mesh_cols: int) -> list[int]:
    """Where each shard of a `ShardTensorToMesh(dim=0)` tensor lands after gathering both mesh axes.

    Returns `order`, where `order[i]` is the index of the **original** shard sitting at position `i` of
    dim 0 in the gathered tensor. So `gathered[i] == original[order[i]]`.

    This exists because the gather is **not** order-preserving (measured: `gathered replica matches
    host: False, maxdiff 7.93`). `ShardTensorToMesh(dim=0)` lays shard `k` on device `k` in row-major
    order, so shard `k` is at mesh position `(k // cols, k % cols)`. Gathering `cluster_axis=0` concatenates each
    mesh *column*'s four shards along dim 0; gathering `cluster_axis=1` then concatenates those
    per-column groups. The result is dim 0 **transposed**: position `c * rows + r` holds shard
    `r * cols + c`.

    Deriving the tile -> position map from this rather than assuming row-major is the point.
    Getting it wrong puts tiles in the wrong place, which the seam gate catches as a spectacular
    failure rather than a subtle one -- but only if something reads the tiles back in the first place.
    """
    return [r * mesh_cols + c for c in range(mesh_cols) for r in range(mesh_rows)]


@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
def test_two_axis_all_gather_permutes_dim0_by_transpose(mesh_device):
    """Pin the permutation empirically, and pin that it is a permutation at all.

    Every shard carries its own index as its value, so the gathered tensor reads out as the permutation
    directly -- no inference, no maxdiff to interpret. Three claims:

    1. the gather really gathered (local dim 0 goes 1 -> 32), so a pass cannot come from a no-op;
    2. the order is exactly `gathered_tile_order`, i.e. a transpose, not row-major;
    3. every device agrees on that order, so a caller may read any single replica.

    Claim 1 matters because a no-op gather reads as a ~39 % timing speedup precisely because it
    moves no data. A test that only compared *sets* of values would pass on a no-op too.
    """
    rows, cols = tuple(mesh_device.shape)
    num_devices = rows * cols
    assert num_devices == mesh_device.get_num_devices(), f"{rows}x{cols} != {mesh_device.get_num_devices()}"

    # Shard k is filled with the value k, so a value identifies its origin uniquely.
    host = torch.arange(num_devices, dtype=torch.float32).reshape(num_devices, 1, 1, 1).expand(num_devices, 1, 1, 32)
    sharded = ttnn.from_torch(
        host.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    assert ttnn.get_device_tensors(sharded)[0].shape[0] == 1, "each device should hold exactly one shard"

    gathered = ttnn.all_gather(sharded, 0, cluster_axis=0, topology=ttnn.Topology.Ring)
    gathered = ttnn.all_gather(gathered, 0, cluster_axis=1, topology=ttnn.Topology.Ring)

    replicas = ttnn.get_device_tensors(gathered)
    # (1) it really gathered.
    assert replicas[0].shape[0] == num_devices, (
        f"local dim 0 is {replicas[0].shape[0]}, expected {num_devices}; the gather did not happen and "
        "any timing from this configuration is measuring a no-op"
    )

    observed = [int(v) for v in ttnn.to_torch(replicas[0])[:, 0, 0, 0].round().tolist()]
    expected = gathered_tile_order(rows, cols)
    logger.info(f"two-axis all-gather order on {rows}x{cols}: {observed}")
    logger.info(f"gathered_tile_order predicts:              {expected}")

    # (2) it is a permutation, and it is the transpose.
    assert sorted(observed) == list(
        range(num_devices)
    ), f"gathered dim 0 is not a permutation of the shards: {observed}"
    assert observed != list(range(num_devices)), (
        "the gather preserved shard order, so the measured permutation no longer exists -- "
        "`gathered_tile_order` is now wrong and must become the identity"
    )
    assert observed == expected, f"order is {observed}, gathered_tile_order predicts {expected}"

    # (3) every device agrees, so reading one replica is legitimate.
    for index, replica in enumerate(replicas[1:], start=1):
        other = [int(v) for v in ttnn.to_torch(replica)[:, 0, 0, 0].round().tolist()]
        assert other == observed, f"device {index} sees order {other}, device 0 sees {observed}"

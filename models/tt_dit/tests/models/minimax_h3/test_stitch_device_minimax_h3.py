# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device tile blend and unpatchify against their host originals, at production geometry.
Separate file because these run FABRIC_1D_RING while test_vae_parallel_minimax_h3.py runs
FABRIC_1D, and fabric_config is a process-global one-shot (second distinct value is TT_FATAL)."""

import pytest
import torch
from loguru import logger

import ttnn

from ....models.vae.minimax_h3.decoder_minimax_h3 import unpatchify
from ....models.vae.minimax_h3.stitch_device_minimax_h3 import DeviceTileStitcher, unpatchify_device
from ....models.vae.minimax_h3.vae_minimax_h3 import MiniMaxH3VaeConfig, split_tiles, stitch_tiles
from ....utils.check import assert_quality

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]

# 1344x768 -> a 4x7 grid with the real derived overlaps ([96,80,80] by height, [80,80,80,80,64,64] by width).
HEIGHT, WIDTH = 768, 1344
PIXEL_FRAMES = 4  # blend does nothing along T; trimmed from the unit's 28 pixel frames for cost
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
    """The whole 4x7 stitch, device against host."""
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

    # Seam bands on their own: a whole-canvas metric averages a boundary defect away.
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


MESH_4X8 = [
    pytest.param(
        (4, 8),
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "require_exact_physical_num_devices": True},
        id="4x8",
    )
]


def gathered_tile_order(mesh_rows: int, mesh_cols: int) -> list[int]:
    """``gathered[i] == original[order[i]]``: the two-axis gather is not order-preserving --
    dim 0 comes back transposed, position ``c * rows + r`` holds shard ``r * cols + c``."""
    return [r * mesh_cols + c for c in range(mesh_cols) for r in range(mesh_rows)]


@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
def test_two_axis_all_gather_permutes_dim0_by_transpose(mesh_device):
    """The gather really gathers, its dim-0 order is ``gathered_tile_order``, and every replica agrees."""
    rows, cols = tuple(mesh_device.shape)
    num_devices = rows * cols
    assert num_devices == mesh_device.get_num_devices(), f"{rows}x{cols} != {mesh_device.get_num_devices()}"

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
    assert replicas[0].shape[0] == num_devices, (
        f"local dim 0 is {replicas[0].shape[0]}, expected {num_devices}; the gather did not happen and "
        "any timing from this configuration is measuring a no-op"
    )

    observed = [int(v) for v in ttnn.to_torch(replicas[0])[:, 0, 0, 0].round().tolist()]
    expected = gathered_tile_order(rows, cols)
    logger.info(f"two-axis all-gather order on {rows}x{cols}: {observed}")
    logger.info(f"gathered_tile_order predicts:              {expected}")

    assert sorted(observed) == list(
        range(num_devices)
    ), f"gathered dim 0 is not a permutation of the shards: {observed}"
    assert observed != list(range(num_devices)), (
        "the gather preserved shard order, so the measured permutation no longer exists -- "
        "`gathered_tile_order` is now wrong and must become the identity"
    )
    assert observed == expected, f"order is {observed}, gathered_tile_order predicts {expected}"

    for index, replica in enumerate(replicas[1:], start=1):
        other = [int(v) for v in ttnn.to_torch(replica)[:, 0, 0, 0].round().tolist()]
        assert other == observed, f"device {index} sees order {other}, device 0 sees {observed}"

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


# --- host-only numerics for the YUV decode path (no device, no fixtures) ---------------------

MINIMAX_H3_PIXEL_MEAN = (0.485, 0.456, 0.406)
MINIMAX_H3_PIXEL_STD = (0.229, 0.224, 0.225)


def test_pixel_denorm_fold_is_exact_and_commutes_with_the_blend():
    """Folding the de-normalization into ``proj_out`` must be exact, not merely close.

    Exactness is what lets the fold sit *before* the tile cross-fade: the blend is a convex
    combination and the fold is affine, so they commute -- and if they did not, every seam would
    carry the error.
    """
    from ....models.vae.minimax_h3.vae_minimax_h3 import MiniMaxH3Vae, MiniMaxH3VaeConfig

    config = MiniMaxH3VaeConfig()
    channels = config.out_channels
    per_channel = config.temporal_compression_ratio * config.spatial_compression_ratio**2
    out_features, in_features = channels * per_channel, 64

    torch.manual_seed(0)
    weight = torch.randn(out_features, in_features, dtype=torch.float64) * 0.1
    bias = torch.randn(out_features, dtype=torch.float64) * 0.1
    hidden = torch.randn(5, in_features, dtype=torch.float64)

    vae = MiniMaxH3Vae(config, mesh_device=None, pixel_denorm=(MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD))
    state = {"proj_out.weight": weight.clone(), "proj_out.bias": bias.clone()}
    vae._fold_pixel_denorm(state)

    mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, dtype=torch.float64).view(1, channels, 1, 1, 1)
    std = torch.tensor(MINIMAX_H3_PIXEL_STD, dtype=torch.float64).view(1, channels, 1, 1, 1)
    patch = (config.temporal_compression_ratio, config.spatial_compression_ratio, config.spatial_compression_ratio)
    reference = (hidden @ weight.T + bias).reshape(5, channels, *patch)
    reference = 2.0 * (reference * std + mean) - 1.0  # pipeline `_denormalize` -> [0,1], then to [-1,1]

    folded = (hidden @ state["proj_out.weight"].T + state["proj_out.bias"]).reshape(5, channels, *patch)
    assert torch.allclose(folded, reference, atol=1e-12), f"max error {(folded - reference).abs().max()}"

    a, b = torch.randn(4, in_features, dtype=torch.float64), torch.randn(4, in_features, dtype=torch.float64)
    w = torch.rand(4, 1, dtype=torch.float64)
    blend_then_project = (w * a + (1 - w) * b) @ state["proj_out.weight"].T + state["proj_out.bias"]
    project_then_blend = w * (a @ state["proj_out.weight"].T + state["proj_out.bias"]) + (1 - w) * (
        b @ state["proj_out.weight"].T + state["proj_out.bias"]
    )
    assert torch.allclose(blend_then_project, project_then_blend, atol=1e-12), "fold does not commute with the blend"


def test_temporal_crossfade_survives_the_yuv_conversion():
    """The chunk cross-fade runs on planar uint8, after the colour conversion, not before it.

    BT.601 is affine and 4:2:0 decimation is linear, so a convex blend commutes with both and only
    the re-rounding costs anything. This pins that cost at 1 LSB; more would mean the reordering is
    unsound and the blend has to move back in front of the conversion.
    """
    import numpy as np

    from ....models.vae.minimax_h3.vae_minimax_h3 import blend_clip_frames

    frames, height, width, extent = 8, 32, 48, 5

    def to_yuv420(rgb):
        r01 = (rgb + 1.0) * 0.5
        r, g, b = r01[:, 0], r01[:, 1], r01[:, 2]
        y = 16.0 + 219.0 * (0.299 * r + 0.587 * g + 0.114 * b)
        cb = 128.0 + 224.0 * (-0.168736 * r - 0.331264 * g + 0.5 * b)
        cr = 128.0 + 224.0 * (0.5 * r - 0.418688 * g - 0.081312 * b)
        sub = lambda p: p.reshape(p.shape[0], p.shape[1] // 2, 2, p.shape[2] // 2, 2).mean(axis=(2, 4))
        q = lambda p: np.clip(np.rint(p), 0, 255).astype(np.uint8)
        planes = [q(y).reshape(frames, -1), q(sub(cb)).reshape(frames, -1), q(sub(cr)).reshape(frames, -1)]
        return np.concatenate(planes, axis=1).reshape(frames, height * 3 // 2, width)

    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, size=(frames, 3, height, width))
    b = rng.uniform(-1, 1, size=(frames, 3, height, width))

    pos = np.arange(extent, dtype=np.float64).reshape(-1, 1, 1, 1)
    rgb_blended = np.concatenate([a[-extent:] * (1 - pos / extent) + b[:extent] * (pos / extent), b[extent:]], axis=0)

    convert_last = to_yuv420(rgb_blended)
    blend_last = blend_clip_frames(to_yuv420(a), to_yuv420(b), extent)

    assert convert_last.shape == blend_last.shape, f"{convert_last.shape} != {blend_last.shape}"
    worst = np.abs(convert_last.astype(int) - blend_last.astype(int)).max()
    assert worst <= 1, f"reordering the cross-fade past the YUV conversion costs {worst} LSB, not <=1"

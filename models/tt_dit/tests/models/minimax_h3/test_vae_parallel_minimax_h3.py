# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device gates for the H/W-sharded MiniMax-H3 encoder.

Three things are being established, separated so a failure names itself:

1. **The reflect edges**, on their own. ``neighbor_pad_async`` has no ``reflect`` mode, so
   the halo pads ``replicate`` and :func:`reflect_edge_correction` repairs the two global
   edges per axis with a per-device 0/1 mask. That correction was verified as host algebra
   but had never executed on device. It has to be gated alone because the error it makes is
   **one pixel of border**: PCC stays high and it reads as a faint vignette, so a
   whole-encoder number would not catch it. Compared elementwise against ``F.pad(mode=
   "reflect")``, exactly, not by PCC.

2. **The asymmetric trailing pad.** H3's downsamplers pre-pad ``(0,1,0,1)`` reflect. Under
   sharding that cannot live in the model -- only the device holding the global bottom/right
   edge may reflect, while interior devices need a real halo row from the neighbour -- so it
   moved into the conv as ``trailing_spatial_padding``, with the halo asymmetric ``(0,1)``.

3. **The whole encoder, sharded vs unsharded.** Sharding is a pure decomposition: the answer
   must not depend on the factor. Comparing sharded against the (already diffusers-gated)
   unsharded result isolates the decomposition from model correctness.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn

from ....models.vae.minimax_h3.conv_minimax_h3 import MiniMaxH3CausalConv3d
from ....models.vae.minimax_h3.decoder_minimax_h3 import MiniMaxH3ViTDecoder3d, unpatchify
from ....models.vae.minimax_h3.encoder_minimax_h3 import MiniMaxH3Encoder3d
from ....models.vae.minimax_h3.stitch_device_minimax_h3 import DeviceTileStitcher, unpatchify_device
from ....models.vae.minimax_h3.vae_minimax_h3 import MiniMaxH3VaeConfig, blend, split_tiles, stitch_tiles
from ....parallel.config import ParallelFactor, VaeHWParallelConfig
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from .test_performance_vae_minimax_h3 import (
    CLIP_FRAMES,
    DECODE_LATENT_FRAMES,
    LATENT_TILE,
    TILE,
    _config,
    _random_decoder_state,
    _random_encoder_state,
    _weights_dir,
)

# One fp32 ulp at unit magnitude (2^-22). The edge correction blends rather than assigns.
FP32_ULP = 1e-6

FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True}

# (mesh, h_factor, w_factor). Height on mesh axis 0 (width 4), width on axis 1 (width 8).
# H3's extents are dyadic so every factor here divides all six levels exactly.
SHARDINGS = [
    pytest.param((4, 8), 4, 1, FABRIC, id="h4"),
    pytest.param((4, 8), 1, 8, FABRIC, id="w8"),
    pytest.param((4, 8), 4, 8, FABRIC, id="h4w8"),
]


def _reflect_pad_spatial(x_BTHWC: torch.Tensor, pad: tuple[int, int, int, int]) -> torch.Tensor:
    """``F.pad(mode="reflect")`` on H/W only, as ``(w_before, w_after, h_before, h_after)``.

    Folded to 4D first: reflect padding of a 5D tensor takes 6 values and would pad T too,
    and H3 pads only the spatial axes (T is causal-zero-padded elsewhere).
    """
    B, T, H, W, C = x_BTHWC.shape
    nchw = x_BTHWC.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
    padded = F.pad(nchw, pad, mode="reflect")
    return padded.reshape(B, T, C, padded.shape[-2], padded.shape[-1]).permute(0, 1, 3, 4, 2)


def _assert_halo_windows(padded, x_BTHWC, mesh_device, h_factor, w_factor, pad_h, pad_w):
    """Each device's padded shard must equal its window of the globally reflect-padded tensor.

    Gathering the padded shards and comparing to the global pad would be wrong: the halos are
    interior duplicates, so four shards of height 8 padded by 1 each side concatenate to 40,
    not the true padded 34. Windowing is also the stronger check -- it pins down *which*
    rows each device received, so an interior halo taken from the wrong neighbour fails, and
    so does a global edge that replicated instead of reflecting.

    Only the sharded axes are padded here: ``_halo_pad`` handles those, while a replicated
    axis keeps its pad local (applied in ``forward``, not here).
    """
    before_h, after_h = pad_h if h_factor > 1 else (0, 0)
    before_w, after_w = pad_w if w_factor > 1 else (0, 0)
    reference = _reflect_pad_spatial(x_BTHWC, (before_w, after_w, before_h, after_h))

    height, width = x_BTHWC.shape[2], x_BTHWC.shape[3]
    local_h, local_w = height // h_factor, width // w_factor
    shards = ttnn.get_device_tensors(padded)
    columns = tuple(mesh_device.shape)[1]
    distinct = h_factor * w_factor

    worst = 0.0
    for i in range(h_factor):
        for j in range(w_factor):
            index = (i * w_factor + j) if len(shards) == distinct else (i * columns + j)
            local = ttnn.to_torch(shards[index]).float()
            window = reference[
                :,
                :,
                i * local_h : i * local_h + local_h + before_h + after_h,
                j * local_w : j * local_w + local_w + before_w + after_w,
                :,
            ]
            assert local.shape == window.shape, f"shard ({i},{j}): {local.shape} != {window.shape}"
            worst = max(worst, (local - window).abs().max().item())
    return worst


def _parallel_config(h_factor: int, w_factor: int) -> VaeHWParallelConfig:
    return VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=h_factor, mesh_axis=0),
        width_parallel=ParallelFactor(factor=w_factor, mesh_axis=1),
    )


def _shard_hw(x_BTHWC: torch.Tensor, mesh_device, h_factor: int, w_factor: int, dtype=ttnn.float32):
    dims = [None, None]
    if h_factor > 1:
        dims[0] = 2
    if w_factor > 1:
        dims[1] = 3
    mapper = (
        ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape))
        if any(d is not None for d in dims)
        else ttnn.ReplicateTensorToMesh(mesh_device)
    )
    return ttnn.from_torch(x_BTHWC, dtype=dtype, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper)


def _gather_hw(x, mesh_device, h_factor: int, w_factor: int) -> torch.Tensor:
    """Reassemble an H/W-sharded tensor on host by reading the shards directly.

    Not ``ConcatMesh2dToTensor``: when one mesh axis is replicated the tensor
    carries only ``h_factor * w_factor`` distinct shards, and the 2D composer requires one
    per mesh coordinate ("ND composition requires the number of tensors 4 to match the mesh
    shape MeshShape([4, 8])"). Indexing the shards is unambiguous and needs no assumption
    about how replication is materialised.
    """
    shards = ttnn.get_device_tensors(x)
    columns = tuple(mesh_device.shape)[1]
    distinct = h_factor * w_factor
    rows = []
    for i in range(h_factor):
        row = []
        for j in range(w_factor):
            # Shards are row-major over the mesh. When an axis is replicated, only the
            # distinct shards are present, so index them linearly instead.
            index = (i * w_factor + j) if len(shards) == distinct else (i * columns + j)
            row.append(ttnn.to_torch(shards[index]).float())
        rows.append(torch.cat(row, dim=3) if len(row) > 1 else row[0])
    return torch.cat(rows, dim=2) if len(rows) > 1 else rows[0]


@pytest.mark.parametrize(
    ("mesh_device", "h_factor", "w_factor", "device_params"), SHARDINGS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("spatial_padding", [1], ids=["pad1"])
def test_reflect_halo_edges_exact(mesh_device, h_factor, w_factor, spatial_padding):
    """The replicate halo + global-edge correction must equal ``F.pad(mode="reflect")`` exactly.

    Runs the pad path alone (no conv) so a border error cannot hide behind a convolution.
    """

    torch.manual_seed(0)
    frames, height, width, channels = 3, 32, 32, 32
    x = torch.randn(1, frames, height, width, channels)

    parallel_config = _parallel_config(h_factor, w_factor)
    ccl = CCLManager(mesh_device=mesh_device, topology=ttnn.Topology.Linear)
    conv = MiniMaxH3CausalConv3d(
        channels,
        channels,
        kernel_size=3,
        spatial_padding=spatial_padding,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl,
    )

    padded = conv._halo_pad(_shard_hw(x, mesh_device, h_factor, w_factor))
    pad = (spatial_padding, spatial_padding)
    worst = _assert_halo_windows(padded, x, mesh_device, h_factor, w_factor, pad, pad)
    logger.info(f"h{h_factor}w{w_factor} pad{spatial_padding}: worst halo element {worst:.3e}")
    # Elementwise, not PCC: the one-pixel border is what is under test. The bound is one fp32
    # ulp rather than zero because the correction is a blend, `t + mask * (s - t)`, which is
    # `s` only in exact arithmetic -- measured 2.384e-07 == 2^-22 across every config. Still
    # four orders tighter than anything PCC would notice.
    assert worst <= FP32_ULP, f"halo differs from reflect by {worst:.3e}"


@pytest.mark.parametrize(
    ("mesh_device", "h_factor", "w_factor", "device_params"), SHARDINGS, indirect=["mesh_device", "device_params"]
)
def test_trailing_reflect_halo_exact(mesh_device, h_factor, w_factor):
    """The downsamplers' asymmetric ``(0,1,0,1)`` reflect pre-pad, sharded."""
    torch.manual_seed(0)
    frames, height, width, channels = 3, 32, 32, 32
    x = torch.randn(1, frames, height, width, channels)

    parallel_config = _parallel_config(h_factor, w_factor)
    ccl = CCLManager(mesh_device=mesh_device, topology=ttnn.Topology.Linear)
    conv = MiniMaxH3CausalConv3d(
        channels,
        channels,
        kernel_size=3,
        stride=(1, 2, 2),
        spatial_padding=0,
        trailing_spatial_padding=1,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl,
    )

    padded = conv._halo_pad(_shard_hw(x, mesh_device, h_factor, w_factor))
    worst = _assert_halo_windows(padded, x, mesh_device, h_factor, w_factor, (0, 1), (0, 1))
    logger.info(f"h{h_factor}w{w_factor} trailing: worst element {worst:.3e}")
    assert worst <= FP32_ULP, f"trailing halo differs from reflect by {worst:.3e}"


@pytest.mark.parametrize(
    ("mesh_device", "h_factor", "w_factor", "device_params"), SHARDINGS, indirect=["mesh_device", "device_params"]
)
def test_encoder_sharded_matches_unsharded(mesh_device, h_factor, w_factor):
    """Sharding is a decomposition: the encoder's answer must not depend on the factor."""
    weights_dir = _weights_dir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    config = _config(weights_dir)
    torch.manual_seed(0)

    state = _random_encoder_state(config)
    common = dict(
        num_frames=CLIP_FRAMES,
        height=TILE,
        width=TILE,
        in_channels=3,
        out_channels=2 * config["latent_channels"],
        block_out_channels=tuple(config["block_out_channels"]),
        layers_per_block=config["layers_per_block"],
        spatial_downsample_factors=tuple(config["spatial_downsample_factors"]),
        temporal_downsample_factors=tuple(config["temporal_downsample_factors"]),
        temporal_taps=3,
        mesh_device=mesh_device,
    )

    reference_encoder = MiniMaxH3Encoder3d(**common)
    reference_encoder.load_torch_state_dict(dict(state))
    x = torch.randn(1, CLIP_FRAMES, TILE, TILE, reference_encoder.conv_in.in_channels)
    x_replicated = ttnn.from_torch(
        x,
        dtype=ttnn.float32,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    expected = ttnn.to_torch(
        reference_encoder(x_replicated), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[:1].float()

    ccl = CCLManager(mesh_device=mesh_device, topology=ttnn.Topology.Linear)
    sharded_encoder = MiniMaxH3Encoder3d(
        **common, parallel_config=_parallel_config(h_factor, w_factor), ccl_manager=ccl
    )
    sharded_encoder.load_torch_state_dict(dict(state))
    actual = _gather_hw(sharded_encoder(_shard_hw(x, mesh_device, h_factor, w_factor)), mesh_device, h_factor, w_factor)

    assert actual.shape == expected.shape, f"{actual.shape} != {expected.shape}"
    logger.info(f"h{h_factor}w{w_factor}: worst element {(actual - expected).abs().max().item():.3e}")
    assert_quality(expected, actual, pcc=0.999)


# -------------------------------------------------------------------- data-parallel independence
#
# Gate: the visual encoder is pure SPMD, so (tile, chunk) work units can be data-parallel.
#
# The encoder's tiling makes every ``(clip, tile)`` unit independent -- 336 of them for
# 768P/5s -- and the module itself contains no CCL: conv3d, GroupNorm3D and the
# elementwise ops are all device-local. So the whole mesh should be usable by handing each
# device a *different* unit and running one identical program, with the weights replicated.
#
# Nothing else in ``tt_dit`` does this. Every existing VAE **replicates** activations and
# shards H/W, so no test anywhere covers "each device holds different data". If any op in
# the stack quietly assumes replicated inputs -- a broadcast, a grid decision taken from a
# global shape, a reduction that spans shards -- the data-parallel scheme is dead, and it
# would show up as a plausible-looking but wrong tile rather than a crash.
#
# The gate is reference-free and therefore cheap: run 32 distinct units data-parallel, then
# re-run selected units **replicated** across all 32 devices. A unit's result must not
# depend on what its neighbours hold, so the replicated result must equal the
# data-parallel one. That also proves the replicas agree with each other, which is the
# same-program-same-answer check. Parity against diffusers is already gated per-unit in
# ``test_vae_minimax_h3.py``; this only has to establish independence.


# No CCL in the encoder, so no fabric: a ring with no traffic still costs the ethernet
# handshake at open time.
MESH_4X8 = [
    pytest.param(
        (4, 8),
        {"fabric_config": None, "require_exact_physical_num_devices": True},
        id="mesh4x8",
    )
]

# Which of the 32 units to re-run replicated. First, last, and one interior device, so a
# contamination that only affects mesh edges or only interior rows is still caught.
PROBE_UNITS = (0, 7, 31)


@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
def test_encoder_data_parallel_independence(mesh_device):
    weights_dir = _weights_dir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    config = _config(weights_dir)
    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(0)

    encoder = MiniMaxH3Encoder3d(
        num_frames=CLIP_FRAMES,
        height=TILE,
        width=TILE,
        in_channels=3,
        out_channels=2 * config["latent_channels"],
        block_out_channels=tuple(config["block_out_channels"]),
        layers_per_block=config["layers_per_block"],
        spatial_downsample_factors=tuple(config["spatial_downsample_factors"]),
        temporal_downsample_factors=tuple(config["temporal_downsample_factors"]),
        temporal_taps=3,
        mesh_device=mesh_device,
    )
    # Random weights: independence is a property of the program, not of the values, and
    # skipping the 10.4 GB checkpoint read is what keeps this gate quick.
    encoder.load_torch_state_dict(_random_encoder_state(config))

    in_channels = encoder.conv_in.in_channels
    units = [torch.randn(1, CLIP_FRAMES, TILE, TILE, in_channels) for _ in range(num_devices)]

    stacked = torch.cat(units, dim=0)
    x_dp = ttnn.from_torch(
        stacked,
        dtype=ttnn.float32,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    # Load-bearing: the per-device shard must be batch 1. If ttnn hands the encoder the
    # *global* 32-unit shape instead, conv3d and GroupNorm3D would size their grids and
    # blockings for 32x the work.
    logger.info(f"host {tuple(stacked.shape)} -> per-device shard {tuple(x_dp.shape)}")
    assert tuple(x_dp.shape) == (1, CLIP_FRAMES, TILE, TILE, in_channels), (
        f"expected a batch-1 per-device shard, got {tuple(x_dp.shape)}; "
        "the encoder's conv3d/GroupNorm are sized from this shape"
    )

    out_dp = ttnn.to_torch(encoder(x_dp), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()
    assert out_dp.shape[0] == num_devices, f"expected {num_devices} gathered units, got {out_dp.shape[0]}"

    for unit in PROBE_UNITS:
        x_rep = ttnn.from_torch(
            units[unit],
            dtype=ttnn.float32,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        out_rep = ttnn.to_torch(encoder(x_rep), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()

        # Same program, same input, 32 devices: the replicas must agree with each other.
        spread = (out_rep - out_rep[0:1]).abs().max().item()
        logger.info(f"unit {unit}: replica spread {spread:.3e}")
        assert spread == 0.0, f"unit {unit}: replicas disagree by {spread:.3e} across devices"

        # And the data-parallel run must reproduce it, i.e. device `unit`'s answer does
        # not depend on the 31 different units its neighbours were holding.
        delta = (out_dp[unit] - out_rep[0]).abs().max().item()
        logger.info(f"unit {unit}: data-parallel vs replicated max abs diff {delta:.3e}")
        assert out_dp[unit].shape == out_rep[0].shape
        assert_quality(out_rep[0], out_dp[unit], pcc=0.999_999)


# Two layers, not 36: independence is a property of the program, and two exercises every op
# the full stack has (fused qkv, the RoPE lane permute, SDPA, swiglu, LayerScale, the
# residual chain) at a eighteenth of the 4.51 GiB weight cost. Whether 36 layers *fit*
# replicated on 32 devices is a separate residency question, answered by the perf run.
DECODER_PROBE_LAYERS = 2


@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
def test_decoder_data_parallel_independence(mesh_device):
    """Same independence gate for the ViT decoder: each (chunk, tile) unit is its own decode."""
    weights_dir = _weights_dir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    config = _config(weights_dir)
    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(0)

    decoder = MiniMaxH3ViTDecoder3d(
        num_frames=DECODE_LATENT_FRAMES,
        height=LATENT_TILE,
        width=LATENT_TILE,
        in_channels=config["latent_channels"],
        out_channels=config["out_channels"],
        num_layers=DECODER_PROBE_LAYERS,
        mesh_device=mesh_device,
    )
    decoder.load_torch_state_dict(_random_decoder_state(config, num_layers=DECODER_PROBE_LAYERS))

    tokens = DECODE_LATENT_FRAMES * LATENT_TILE * LATENT_TILE
    units = [torch.randn(1, tokens, config["latent_channels"]) for _ in range(num_devices)]

    x_dp = ttnn.from_torch(
        torch.cat(units, dim=0),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    logger.info(f"decoder per-device shard {tuple(x_dp.shape)}")
    assert tuple(x_dp.shape) == (1, tokens, config["latent_channels"])

    out_dp = ttnn.to_torch(decoder(x_dp), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()
    assert out_dp.shape[0] == num_devices

    for unit in PROBE_UNITS:
        x_rep = ttnn.from_torch(
            units[unit],
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        out_rep = ttnn.to_torch(decoder(x_rep), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()

        spread = (out_rep - out_rep[0:1]).abs().max().item()
        logger.info(f"decoder unit {unit}: replica spread {spread:.3e}")
        assert spread == 0.0, f"unit {unit}: replicas disagree by {spread:.3e}"

        delta = (out_dp[unit] - out_rep[0]).abs().max().item()
        logger.info(f"decoder unit {unit}: data-parallel vs replicated max abs diff {delta:.3e}")
        assert out_dp[unit].shape == out_rep[0].shape
        assert_quality(out_rep[0], out_dp[unit], pcc=0.999_999)


# -------------------------------------------------------------------- the device tile blend and unpatchify
#
# The device tile blend and unpatchify against their host originals, at the production geometry.
#
# This is the piece that has to be right before any of it is worth doing. The reference blend is
# sequential and asymmetric; a separable reformulation was measured to move 11.1 % of pixels by up to
# 4.66, so the device version mirrors the order rather than the algebra. These tests are what say it
# did.
#
# `single_device` throughout: the question here is the arithmetic, not the distribution. The
# all-gather that co-locates neighbouring tiles is a separate concern and gated separately.


SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]

# 1344x768 with the real tile size and overlap: a 4x7 grid, overlaps [96, 80, 80] by height and
# [80, 80, 80, 80, 64, 64] by width. Derived rather than hardcoded -- hardcoding an overlap of 32
# here is exactly the mistake that made an earlier seam gate check non-boundary columns.
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
@pytest.mark.parametrize("dim", [-2, -1], ids=["height", "width"])
def test_blend_matches_host(mesh_device, dim, reset_seeds):
    """One cross-fade, device against host, at a real overlap extent."""
    height_overlaps, width_overlaps = _geometry()
    extent = height_overlaps[0] if dim == -2 else width_overlaps[0]

    a = torch.randn(1, CHANNELS, PIXEL_FRAMES, 256, 256)
    b = torch.randn(1, CHANNELS, PIXEL_FRAMES, 256, 256)
    expected = blend(a, b, extent, dim=dim)

    stitcher = DeviceTileStitcher(mesh_device)
    to_dev = lambda x: ttnn.from_torch(x, dtype=ttnn.float32, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    actual = ttnn.to_torch(stitcher.blend(to_dev(a), to_dev(b), extent, dim=dim))

    assert actual.shape == expected.shape, f"{tuple(actual.shape)} != {tuple(expected.shape)}"
    logger.info(f"blend dim={dim} extent={extent}: {tuple(expected.shape)}")
    assert_quality(expected, actual, pcc=0.9999)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_stitch_matches_host_at_production_geometry(mesh_device, reset_seeds):
    """The whole 4x7 stitch, device against host.

    The bar is `pcc=0.9999` with `relative_rmse` paired: the blend output is consumed as an absolute
    pixel value, and a seam is a *local* defect that a whole-canvas PCC can dilute -- so the seam
    columns are also checked on their own below.
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

    This exists because the gather is **not** order-preserving, which STATE.md amendment 84 measured
    the hard way (`gathered replica matches host: False, maxdiff 7.93`) and then left unpinned as its
    first loose end. `ShardTensorToMesh(dim=0)` lays shard `k` on device `k` in row-major order, so
    shard `k` is at mesh position `(k // cols, k % cols)`. Gathering `cluster_axis=0` concatenates each
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

    Claim 1 matters because amendment 86's readback bug looked like a 39 % speedup precisely because it
    moved no data. A test that only compared *sets* of values would pass on a no-op too.
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
        "the gather preserved shard order, so amendment 84's permutation no longer exists -- "
        "`gathered_tile_order` is now wrong and must become the identity"
    )
    assert observed == expected, f"order is {observed}, gathered_tile_order predicts {expected}"

    # (3) every device agrees, so reading one replica is legitimate.
    for index, replica in enumerate(replicas[1:], start=1):
        other = [int(v) for v in ttnn.to_torch(replica)[:, 0, 0, 0].round().tolist()]
        assert other == observed, f"device {index} sees order {other}, device 0 sees {observed}"

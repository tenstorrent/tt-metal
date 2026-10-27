# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device gates for the H/W-sharded MiniMax-H3 encoder: reflect halo edges, the asymmetric
trailing pad, sharded-vs-unsharded parity, and data-parallel unit independence. fabric_config
is a process-global one-shot, so the FABRIC_1D_RING gates live in test_stitch_device_minimax_h3.py."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn

from ....models.vae.minimax_h3.conv_minimax_h3 import MiniMaxH3CausalConv3d
from ....parallel.config import ParallelFactor, VaeHWParallelConfig
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from .common import (
    CLIP_FRAMES,
    DECODE_LATENT_FRAMES,
    LATENT_TILE,
    TILE,
    build_visual_decoder,
    build_visual_encoder,
    load_config,
    random_decoder_state,
    random_encoder_state,
    weights_subdir,
)

FP32_ULP = 1e-6  # one fp32 ulp at unit magnitude (2^-22); the edge correction blends rather than assigns

FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True}

# Full h4w8 only: exercises halo exchange and edge correction on both axes, subsuming h4 / w8.
SHARDINGS = [
    pytest.param((4, 8), 4, 8, FABRIC, id="h4w8"),
]


def _reflect_pad_spatial(x_BTHWC: torch.Tensor, pad: tuple[int, int, int, int]) -> torch.Tensor:
    """``F.pad(mode="reflect")`` on H/W only, as ``(w_before, w_after, h_before, h_after)``."""
    B, T, H, W, C = x_BTHWC.shape
    nchw = x_BTHWC.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
    padded = F.pad(nchw, pad, mode="reflect")
    return padded.reshape(B, T, C, padded.shape[-2], padded.shape[-1]).permute(0, 1, 3, 4, 2)


def _assert_halo_windows(padded, x_BTHWC, mesh_device, h_factor, w_factor, pad_h, pad_w):
    """Each device's padded shard must equal its window of the globally reflect-padded tensor."""
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
    """Reassemble by indexing shards; ConcatMesh2dToTensor requires one shard per mesh coordinate."""
    shards = ttnn.get_device_tensors(x)
    columns = tuple(mesh_device.shape)[1]
    distinct = h_factor * w_factor
    rows = []
    for i in range(h_factor):
        row = []
        for j in range(w_factor):
            index = (i * w_factor + j) if len(shards) == distinct else (i * columns + j)
            row.append(ttnn.to_torch(shards[index]).float())
        rows.append(torch.cat(row, dim=3) if len(row) > 1 else row[0])
    return torch.cat(rows, dim=2) if len(rows) > 1 else rows[0]


@pytest.mark.parametrize(
    ("mesh_device", "h_factor", "w_factor", "device_params"), SHARDINGS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize(
    "pad_kind",
    [
        pytest.param("symmetric", id="pad1"),
        pytest.param("trailing", id="trailing"),
    ],
)
def test_reflect_halo_edges_exact(mesh_device, h_factor, w_factor, pad_kind):
    """The replicate halo + global-edge correction must equal ``F.pad(mode="reflect")`` exactly."""
    torch.manual_seed(0)
    frames, height, width, channels = 3, 32, 32, 32
    x = torch.randn(1, frames, height, width, channels)

    parallel_config = _parallel_config(h_factor, w_factor)
    ccl = CCLManager(mesh_device=mesh_device, topology=ttnn.Topology.Linear)
    if pad_kind == "symmetric":
        conv = MiniMaxH3CausalConv3d(
            channels,
            channels,
            kernel_size=3,
            spatial_padding=1,
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl,
        )
        pad = (1, 1)
    else:
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
        pad = (0, 1)

    padded = conv._halo_pad(_shard_hw(x, mesh_device, h_factor, w_factor))
    worst = _assert_halo_windows(padded, x, mesh_device, h_factor, w_factor, pad, pad)
    logger.info(f"h{h_factor}w{w_factor} {pad_kind}: worst halo element {worst:.3e}")
    assert worst <= FP32_ULP, f"halo differs from reflect by {worst:.3e}"  # measured 2.384e-07 == 2^-22


@pytest.mark.parametrize(
    ("mesh_device", "h_factor", "w_factor", "device_params"), SHARDINGS, indirect=["mesh_device", "device_params"]
)
def test_encoder_sharded_matches_unsharded(mesh_device, h_factor, w_factor):
    """Sharding is a decomposition: the encoder's answer must not depend on the factor."""
    weights_dir = weights_subdir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_MODEL_PATH")
    config = load_config(weights_dir)
    torch.manual_seed(0)

    state = random_encoder_state(config)

    reference_encoder = build_visual_encoder(config, mesh_device, CLIP_FRAMES, temporal_taps=3)
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
    sharded_encoder = build_visual_encoder(
        config,
        mesh_device,
        CLIP_FRAMES,
        temporal_taps=3,
        parallel_config=_parallel_config(h_factor, w_factor),
        ccl_manager=ccl,
    )
    sharded_encoder.load_torch_state_dict(dict(state))
    actual = _gather_hw(sharded_encoder(_shard_hw(x, mesh_device, h_factor, w_factor)), mesh_device, h_factor, w_factor)

    assert actual.shape == expected.shape, f"{actual.shape} != {expected.shape}"
    logger.info(f"h{h_factor}w{w_factor}: worst element {(actual - expected).abs().max().item():.3e}")
    assert_quality(expected, actual, pcc=0.999)


# -------------------------------------------------------------------- data-parallel independence

# No CCL in the encoder, so no fabric: a ring with no traffic still costs the ethernet
# handshake at open time.
MESH_4X8 = [
    pytest.param(
        (4, 8),
        {"fabric_config": None, "require_exact_physical_num_devices": True},
        id="mesh4x8",
    )
]

PROBE_UNITS = (0, 7, 31)  # first, last, one interior device


def _assert_unit_independence(module, units, out_dp, mesh_device, *, dtype, layout, label="unit"):
    """Re-run each probe unit replicated across the mesh and hold the data-parallel run to it."""
    for unit in PROBE_UNITS:
        x_rep = ttnn.from_torch(
            units[unit],
            dtype=dtype,
            device=mesh_device,
            layout=layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        out_rep = ttnn.to_torch(module(x_rep), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()

        spread = (out_rep - out_rep[0:1]).abs().max().item()
        logger.info(f"{label} {unit}: replica spread {spread:.3e}")
        assert spread == 0.0, f"unit {unit}: replicas disagree by {spread:.3e} across devices"

        delta = (out_dp[unit] - out_rep[0]).abs().max().item()
        logger.info(f"{label} {unit}: data-parallel vs replicated max abs diff {delta:.3e}")
        assert out_dp[unit].shape == out_rep[0].shape
        assert_quality(out_rep[0], out_dp[unit], pcc=0.999_999)


@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
def test_encoder_data_parallel_independence(mesh_device):
    weights_dir = weights_subdir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_MODEL_PATH")
    config = load_config(weights_dir)
    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(0)

    encoder = build_visual_encoder(config, mesh_device, CLIP_FRAMES, temporal_taps=3)
    encoder.load_torch_state_dict(random_encoder_state(config))

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
    # Per-device shard must be batch 1: the global shape would size conv3d/GroupNorm grids for 32x the work.
    logger.info(f"host {tuple(stacked.shape)} -> per-device shard {tuple(x_dp.shape)}")
    assert tuple(x_dp.shape) == (1, CLIP_FRAMES, TILE, TILE, in_channels), (
        f"expected a batch-1 per-device shard, got {tuple(x_dp.shape)}; "
        "the encoder's conv3d/GroupNorm are sized from this shape"
    )

    out_dp = ttnn.to_torch(encoder(x_dp), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()
    assert out_dp.shape[0] == num_devices, f"expected {num_devices} gathered units, got {out_dp.shape[0]}"

    _assert_unit_independence(encoder, units, out_dp, mesh_device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)


DECODER_PROBE_LAYERS = 2  # two layers exercise every op; 36-layer residency is a perf question


@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
def test_decoder_data_parallel_independence(mesh_device):
    """Same independence gate for the ViT decoder: each (chunk, tile) unit is its own decode."""
    weights_dir = weights_subdir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_MODEL_PATH")
    config = load_config(weights_dir)
    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(0)

    decoder = build_visual_decoder(config, mesh_device, num_layers=DECODER_PROBE_LAYERS)
    decoder.load_torch_state_dict(random_decoder_state(config, num_layers=DECODER_PROBE_LAYERS))

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

    _assert_unit_independence(
        decoder, units, out_dp, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, label="decoder unit"
    )

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gate: the visual encoder is pure SPMD, so (tile, chunk) work units can be data-parallel.

The encoder's tiling makes every ``(clip, tile)`` unit independent -- 336 of them for
768P/5s -- and the module itself contains no CCL: conv3d, GroupNorm3D and the
elementwise ops are all device-local. So the whole mesh should be usable by handing each
device a *different* unit and running one identical program, with the weights replicated.

Nothing else in ``tt_dit`` does this. Every existing VAE **replicates** activations and
shards H/W, so no test anywhere covers "each device holds different data". If any op in
the stack quietly assumes replicated inputs -- a broadcast, a grid decision taken from a
global shape, a reduction that spans shards -- the data-parallel scheme is dead, and it
would show up as a plausible-looking but wrong tile rather than a crash.

The gate is reference-free and therefore cheap: run 32 distinct units data-parallel, then
re-run selected units **replicated** across all 32 devices. A unit's result must not
depend on what its neighbours hold, so the replicated result must equal the
data-parallel one. That also proves the replicas agree with each other, which is the
same-program-same-answer check. Parity against diffusers is already gated per-unit in
``test_vae_encoder_minimax_h3.py``; this only has to establish independence.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn

from ....models.vae.minimax_h3.decoder_minimax_h3 import MiniMaxH3ViTDecoder3d
from ....models.vae.minimax_h3.encoder_minimax_h3 import MiniMaxH3Encoder3d
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

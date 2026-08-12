# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device gates for the H/W-sharded MiniMax-H3 encoder.

**One fabric config per file.** ``fabric_config`` is a process-global one-shot
(``tt_metal/impl/context/metal_env.cpp:293``), not a per-test fixture: the second *distinct*
non-None value in a process raises ``TT_FATAL: Tried to override previous value of fabric
config``. So the ``FABRIC_1D_RING`` stitch gates live in
``test_stitch_device_minimax_h3.py``, not here, and this file stays ``FABRIC_1D`` only.

Three things are being established, separated so a failure names itself:

1. **The reflect edges**, on their own. ``neighbor_pad_async`` has no ``reflect`` mode, so
   the halo pads ``replicate`` and :func:`reflect_edge_correction` repairs the two global
   edges per axis with a per-device 0/1 mask. The correction is gated alone because the
   error it makes is **one pixel of border**: PCC stays high and it reads as a faint
   vignette, so a whole-encoder number would not catch it. Compared elementwise against
   ``F.pad(mode="reflect")``, exactly, not by PCC.

2. **The asymmetric trailing pad.** H3's downsamplers pre-pad ``(0,1,0,1)`` reflect. Under
   sharding that cannot live in the model -- only the device holding the global bottom/right
   edge may reflect, while interior devices need a real halo row from the neighbour -- so it
   lives in the conv as ``trailing_spatial_padding``, with the halo asymmetric ``(0,1)``.

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

# One fp32 ulp at unit magnitude (2^-22). The edge correction blends rather than assigns.
FP32_ULP = 1e-6

FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True}

# (mesh, h_factor, w_factor). Height on mesh axis 0 (width 4), width on axis 1 (width 8).
# H3's extents are dyadic so every factor here divides all six levels exactly. Only the
# full h4w8 sharding runs: it exercises the halo exchange and the global-edge correction on
# both axes at once, subsuming the single-axis h4 / w8 cases.
SHARDINGS = [
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
@pytest.mark.parametrize(
    "pad_kind",
    [
        # The resnet convs' symmetric pad-1 reflect, and the downsamplers' asymmetric
        # (0,1,0,1) reflect pre-pad, carried as the conv's trailing_spatial_padding.
        pytest.param("symmetric", id="pad1"),
        pytest.param("trailing", id="trailing"),
    ],
)
def test_reflect_halo_edges_exact(mesh_device, h_factor, w_factor, pad_kind):
    """The replicate halo + global-edge correction must equal ``F.pad(mode="reflect")`` exactly.

    Runs the pad path alone (no conv) so a border error cannot hide behind a convolution.
    ``trailing`` is the downsamplers' asymmetric ``(0,1,0,1)`` pre-pad, sharded.
    """

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
    # Elementwise, not PCC: the one-pixel border is what is under test. The bound is one fp32
    # ulp rather than zero because the correction is a blend, `t + mask * (s - t)`, which is
    # `s` only in exact arithmetic -- measured 2.384e-07 == 2^-22 across every config. Still
    # four orders tighter than anything PCC would notice.
    assert worst <= FP32_ULP, f"halo differs from reflect by {worst:.3e}"


@pytest.mark.parametrize(
    ("mesh_device", "h_factor", "w_factor", "device_params"), SHARDINGS, indirect=["mesh_device", "device_params"]
)
def test_encoder_sharded_matches_unsharded(mesh_device, h_factor, w_factor):
    """Sharding is a decomposition: the encoder's answer must not depend on the factor."""
    weights_dir = weights_subdir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
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


def _assert_unit_independence(module, units, out_dp, mesh_device, *, dtype, layout, label="unit"):
    """Re-run each probe unit replicated across the mesh and hold the data-parallel run to it.

    A unit's result must not depend on what its neighbours hold, so the replicated result
    must equal the data-parallel one -- and the replicas must first agree with each other,
    which is the same-program-same-answer check. The per-unit logging stays inside so a
    failure reads the same as it always has.
    """
    for unit in PROBE_UNITS:
        x_rep = ttnn.from_torch(
            units[unit],
            dtype=dtype,
            device=mesh_device,
            layout=layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        out_rep = ttnn.to_torch(module(x_rep), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()

        # Same program, same input, 32 devices: the replicas must agree with each other.
        spread = (out_rep - out_rep[0:1]).abs().max().item()
        logger.info(f"{label} {unit}: replica spread {spread:.3e}")
        assert spread == 0.0, f"unit {unit}: replicas disagree by {spread:.3e} across devices"

        # And the data-parallel run must reproduce it, i.e. device `unit`'s answer does
        # not depend on the different units its neighbours were holding.
        delta = (out_dp[unit] - out_rep[0]).abs().max().item()
        logger.info(f"{label} {unit}: data-parallel vs replicated max abs diff {delta:.3e}")
        assert out_dp[unit].shape == out_rep[0].shape
        assert_quality(out_rep[0], out_dp[unit], pcc=0.999_999)


@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
def test_encoder_data_parallel_independence(mesh_device):
    weights_dir = weights_subdir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    config = load_config(weights_dir)
    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(0)

    encoder = build_visual_encoder(config, mesh_device, CLIP_FRAMES, temporal_taps=3)
    # Random weights: independence is a property of the program, not of the values, and
    # skipping the 10.4 GB checkpoint read is what keeps this gate quick.
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

    _assert_unit_independence(encoder, units, out_dp, mesh_device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)


# Two layers, not 36: independence is a property of the program, and two exercises every op
# the full stack has (fused qkv, the RoPE lane permute, SDPA, swiglu, LayerScale, the
# residual chain) at a eighteenth of the 4.51 GiB weight cost. Whether 36 layers *fit*
# replicated on 32 devices is a separate residency question, answered by the perf run.
DECODER_PROBE_LAYERS = 2


@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
def test_decoder_data_parallel_independence(mesh_device):
    """Same independence gate for the ViT decoder: each (chunk, tile) unit is its own decode."""
    weights_dir = weights_subdir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
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

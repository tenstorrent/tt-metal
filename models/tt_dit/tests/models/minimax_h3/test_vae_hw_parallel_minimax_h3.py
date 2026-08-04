# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device gates for the H/W-sharded MiniMax-H3 encoder.

Three things are being established, deliberately separated so a failure names itself:

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
from ....models.vae.minimax_h3.encoder_minimax_h3 import MiniMaxH3Encoder3d
from ....parallel.config import ParallelFactor, VaeHWParallelConfig
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from .test_performance_vae_minimax_h3 import CLIP_FRAMES, TILE, _config, _random_encoder_state, _weights_dir

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

    Deliberately not ``ConcatMesh2dToTensor``: when one mesh axis is replicated the tensor
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
    # Elementwise, not PCC: the whole point is the one-pixel border. The bound is one fp32
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

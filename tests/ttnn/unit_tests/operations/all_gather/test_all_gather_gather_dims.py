# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 — non-contiguous concat addressing (gather_dim -3, -2, -1).

Phase 0 / Refinement 1 proved gather_dim=-4 (dim 0), where a device's slice is a
contiguous output page range. For gather_dim != 0 the slice is INTERLEAVED in the
output (op_design "Dataflow Strategy" concat-stride table). This file exercises the
non-contiguous walk directly with the full-gather oracle:

  build full -> ShardTensorToMesh(dim=gather_axis) -> all_gather -> assert EVERY
  device now holds the full concatenated tensor.

Two addressing regimes are covered:
  * whole-page remap (TILE any aligned axis; ROW_MAJOR non-innermost) — the input
    page maps to a strided output page out_p = (in_p//B)*B*N + (in_p%B) + j*B;
  * sub-page byte concat (ROW_MAJOR + innermost gather_dim=-1) — the concat lives
    WITHIN a row, at byte offset j*input_page_size in an N x larger output page.

Structural gap (EXCLUSIONS): TILE + gather along a NON-tile-aligned axis needs a
sub-tile untilize/re-tilize this pure-byte-movement op cannot do — verified rejected
below. Runs on the WH (1,8) FABRIC_1D sim via run_multidevice_sim_pytest.py --op all_gather.
"""

from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.all_gather import all_gather

PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}
LINEAR = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


def _run_full_gather(mesh_device, shard_shape, gather_dim, dtype, layout):
    """Full-gather oracle: shard a seeded full tensor along the gather axis, gather, and
    assert every device holds the full concatenated tensor."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("all_gather requires at least 2 mesh devices")
    rank = len(shard_shape)
    axis = (gather_dim if gather_dim < 0 else gather_dim - rank) % rank

    full_shape = list(shard_shape)
    full_shape[axis] *= num_devices
    torch.manual_seed(7)
    torch_full = torch.randn(tuple(full_shape), dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        torch_full = torch_full.to(torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=axis),
    )
    ttnn.synchronize_device(mesh_device)

    output_tensor = all_gather(input_tensor, gather_dim, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
    ref = torch_full.to(torch.float32)
    for dev_idx, dev_out in enumerate(output_shards):
        assert tuple(dev_out.shape) == tuple(
            full_shape
        ), f"device {dev_idx} shape {tuple(dev_out.shape)} != full {tuple(full_shape)}"
        assert_with_pcc(ref, dev_out.to(torch.float32), PCC[dtype])
    logger.info(
        f"all_gather gd={gather_dim} {dtype} {layout} shard={shard_shape}: all {num_devices} devices hold the full tensor"
    )


# (dtype, layout, gather_dim, shard_shape) — representative non-contiguous cells.
# whole-page remap + sub-page RM innermost + outer-stride (multi-batch) + bf8b + f32.
NONZERO_CELLS = [
    # gather_dim=-1 (innermost W)
    (ttnn.bfloat16, ttnn.TILE_LAYOUT, -1, (1, 1, 64, 128)),  # row-stride: input_Wt tiles per row, stride output_Wt
    (ttnn.float32, ttnn.TILE_LAYOUT, -1, (1, 1, 32, 96)),  # wider W, f32
    (ttnn.bfloat8_b, ttnn.TILE_LAYOUT, -1, (1, 1, 64, 128)),  # block-float whole-tile move
    (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, -1, (1, 1, 32, 32)),  # SUB-PAGE byte concat
    (ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, -1, (1, 1, 48, 64)),  # SUB-PAGE, non-aligned H (RM: no tiling)
    # gather_dim=-2 (H)
    (ttnn.bfloat16, ttnn.TILE_LAYOUT, -2, (1, 1, 96, 64)),  # aligned multi-tile H, contiguous per-outer
    (ttnn.float32, ttnn.TILE_LAYOUT, -2, (1, 1, 32, 32)),  # single tile-row per slice
    (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, -2, (1, 1, 48, 64)),  # non-aligned H, RM whole-row remap (OK)
    # gather_dim=-3 (C)
    (ttnn.bfloat16, ttnn.TILE_LAYOUT, -3, (2, 1, 32, 64)),  # outer stride (d0=2 -> two strided blocks)
    (ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, -3, (1, 1, 32, 32)),  # RM non-innermost whole-page remap
]


@pytest.mark.parametrize("device_params", [LINEAR], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("dtype, layout, gather_dim, shard_shape", NONZERO_CELLS)
def test_all_gather_nonzero_gather_dim(mesh_device, dtype, layout, gather_dim, shard_shape):
    """Non-contiguous concat: every device holds the full tensor for gather_dim in {-3,-2,-1}."""
    _run_full_gather(mesh_device, shard_shape, gather_dim, dtype, layout)


@pytest.mark.parametrize("device_params", [LINEAR], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b])
def test_all_gather_tile_nonaligned_axis_rejected(mesh_device, dtype):
    """TILE + gather along a NON-tile-aligned axis (H=48, gather_dim=-2) is an EXCLUSIONS
    structural gap (sub-tile repack unsupported): validate() must refuse it (NotImplementedError
    subclass) BEFORE any device work — not silently produce wrong output."""
    if prod(tuple(mesh_device.shape)) < 2:
        pytest.skip("all_gather requires at least 2 mesh devices")
    shard_shape = (1, 1, 48, 64)  # H=48 is non-tile-aligned
    full_shape = (1, 1, 48 * prod(tuple(mesh_device.shape)), 64)
    torch_full = torch.randn(full_shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        torch_full = torch_full.to(torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )
    ttnn.synchronize_device(mesh_device)
    with pytest.raises(NotImplementedError):
        all_gather(input_tensor, -2, topology=ttnn.Topology.Linear)
    logger.info(f"excluded cell TILE {dtype} gd=-2 non_tile_aligned correctly refused")

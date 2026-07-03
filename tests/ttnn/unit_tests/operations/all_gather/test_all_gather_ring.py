# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Refinement 3 — Ring topology coverage for the all_gather CCL op.

all_gather's bidirectional store-and-forward is topology-agnostic at the routing
level: every fabric hop is between IMMEDIATE neighbours (1 hop), and ccl_dm_route
returns the SAME route for Ring and Linear on an adjacent pair. So the per-device
program is byte-identical for Ring and Linear, and the op produces a correct
all-gather on BOTH fabric planes:

  * FABRIC_1D      (the line fabric the golden suite / acceptance test use) —
    Ring here proves the "bidirectional design also runs on a Ring" claim
    (op_design.md "Ring algorithm"): topology=Ring is accepted and gathers
    correctly on the line fabric, since the store-and-forward never emits a
    >1-hop (wraparound) route.
  * FABRIC_1D_RING (a genuine ring fabric with the wraparound link) — proves the
    op runs correctly on the real ring data plane the T3K 1x8 RING mesh-graph
    descriptor + FABRIC_1D_RING provisions.

The op does NOT (yet) exploit the wraparound link for a shorter single-direction
gather — that is a perf-only follow-up (op_requirements Refinement 3b), NOT a
correctness gap. Both fabric planes give bit-exact identity gather.

Run on the deterministic WH multi-device sim (mesh (1, 8)):

    scripts/run_multidevice_sim_pytest.py --op all_gather -- \
        tests/ttnn/unit_tests/operations/all_gather/test_all_gather_ring.py -v
"""

from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.all_gather import all_gather


# PCC tolerances keyed by dtype (same thresholds as the acceptance suite).
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

# The two 1-D fabric planes a Ring op can legitimately run on. FABRIC_1D is the
# line fabric the golden suite pins; FABRIC_1D_RING is the genuine ring fabric
# (wraparound link enabled). all_gather's bidirectional store-and-forward gathers
# correctly on both because it only ever emits 1-hop (adjacent) routes.
FABRIC_LINE = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}
FABRIC_RING = {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}

# Per-device (tile-aligned) shard shapes across gather axes. dim 0 is the primary
# proven gather axis; the multi-tile / non-square shapes exercise the relay walk.
SHARD_SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 64, 128),  # multi-tile
    (2, 1, 32, 64),  # multi-batch (gather_dim=0 grows the batch axis)
]


def _make_sharded_input(mesh_device, shard_shape, dtype, layout, shard_axis=0):
    """Shard a freshly-seeded full tensor along ``shard_axis`` across the line.

    Returns the ttnn input tensor and the torch full tensor (the all_gather
    oracle: every device must end up holding this full tensor)."""
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = list(shard_shape)
    full_shape[shard_axis] *= num_devices

    torch.manual_seed(42)
    torch_full = torch.randn(tuple(full_shape), dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        torch_full = torch_full.to(torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=shard_axis),
    )
    ttnn.synchronize_device(mesh_device)
    return input_tensor, torch_full


def _run_ring_gather(mesh_device, shard_shape, dtype, layout, gather_dim):
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("all_gather requires at least 2 mesh devices")

    rank = len(shard_shape)
    shard_axis = (gather_dim if gather_dim < 0 else gather_dim - rank) % rank
    input_tensor, torch_full = _make_sharded_input(mesh_device, shard_shape, dtype, layout, shard_axis)

    output_tensor = all_gather(input_tensor, gather_dim, topology=ttnn.Topology.Ring)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
    torch_ref = torch_full.to(torch.float32)
    for dev_idx, dev_out in enumerate(output_shards):
        assert tuple(dev_out.shape) == tuple(
            torch_full.shape
        ), f"device {dev_idx} output shape {tuple(dev_out.shape)} != full {tuple(torch_full.shape)}"
        assert_with_pcc(torch_ref, dev_out.to(torch.float32), PCC[dtype])
    logger.info(
        f"all_gather Ring {dtype} {layout} shard={shard_shape} gd={gather_dim} on {num_devices} devices: "
        "every device holds the full tensor"
    )


# ---------------------------------------------------------------------------
# Ring on the LINE fabric (FABRIC_1D) — matches the golden suite's fabric plane.
# Proves topology=Ring is accepted and gathers correctly on a line fabric.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [FABRIC_LINE], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES)
def test_all_gather_ring_on_line_fabric(mesh_device, dtype, shard_shape):
    """topology=Ring on FABRIC_1D gathers correctly (bidirectional-line-on-ring)."""
    _run_ring_gather(mesh_device, shard_shape, dtype, ttnn.TILE_LAYOUT, gather_dim=0)


# ---------------------------------------------------------------------------
# Ring on the RING fabric (FABRIC_1D_RING) — the genuine ring data plane.
# Proves the op runs correctly on the wraparound-enabled fabric.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [FABRIC_RING], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES)
def test_all_gather_ring_on_ring_fabric(mesh_device, dtype, shard_shape):
    """topology=Ring on FABRIC_1D_RING gathers correctly (genuine ring fabric)."""
    _run_ring_gather(mesh_device, shard_shape, dtype, ttnn.TILE_LAYOUT, gather_dim=0)


# ---------------------------------------------------------------------------
# Ring across non-contiguous gather axes (Refinement 2 walk) on the ring fabric.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [FABRIC_RING], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("gather_dim", [-3, -2, -1])
def test_all_gather_ring_gather_dims(mesh_device, gather_dim):
    """topology=Ring composes with the non-contiguous concat walk (gather_dim -3/-2/-1)."""
    _run_ring_gather(mesh_device, (1, 1, 64, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT, gather_dim=gather_dim)

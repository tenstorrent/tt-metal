# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for the self-contained Python all_gather CCL op.

all_gather gathers each device's shard of a mesh-sharded tensor and concatenates
all shards along ``gather_dim`` so that AFTER the op EVERY participating device
holds the full concatenated tensor (identical on every device). It is pure data
movement (identity gather, no arithmetic), so the oracle is identity:

  * each device's output == the host-side concatenation along ``gather_dim`` of
    all N devices' input shards == the original (pre-shard) full tensor.

This file is the immutable spec — the implementer must not modify it. It pins
the PROVEN milestone (gather_dim=0, TILE_LAYOUT, bfloat16/float32, Linear). The
broader RM / Ring / non-zero-gather_dim ambition lives in the golden suite
(``eval/golden_tests/all_gather/feature_spec.py``).

Verification topology (MUST match the sim's fixed mesh-graph descriptor):
a Wormhole T3K **line mesh of shape (1, 8)** with
``fabric_config = ttnn.FabricConfig.FABRIC_1D``, driven by
``scripts/run_multidevice_sim_pytest.py --op all_gather``. A different mesh
shape hangs fabric init ("Fabric Router Sync: Timeout") — a test/topology
mismatch, not an op defect.
"""

from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.all_gather import all_gather


# PCC tolerances keyed by dtype (same thresholds as the golden suite).
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

# Proven (dtype, layout) pairs. TILE_LAYOUT is the proven layout; bfloat16 is
# primary, float32 is the second required dtype.
DTYPE_LAYOUTS = [
    (ttnn.bfloat16, ttnn.TILE_LAYOUT),
    (ttnn.float32, ttnn.TILE_LAYOUT),
]

# Per-device (tile-aligned) shard shapes: single-tile, multi-tile, non-square,
# multi-batch. dim 0 is the gather axis; the full tensor's dim 0 is this scaled
# by the device count.
SHARD_SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 64, 128),  # multi-tile
    (1, 1, 96, 64),  # non-square, tile-aligned
    (2, 1, 32, 64),  # multi-batch (dim 0 = 2)
]

# The sim's all_gather topology is a FABRIC_1D line.
LINEAR = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


def _make_sharded_input(mesh_device, shard_shape, dtype, layout):
    """Shard a freshly-seeded full tensor along dim 0 across the whole line.

    The full tensor has shape ``(N * shard_shape[0], *shard_shape[1:])``; each
    device receives exactly ``shard_shape``. Returns the ttnn input tensor and
    the torch full tensor (the all_gather oracle for gather_dim=0).
    """
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])

    torch.manual_seed(42)
    torch_full = torch.randn(full_shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        torch_full = torch_full.to(torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    return input_tensor, torch_full


@pytest.mark.parametrize("device_params", [LINEAR], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("dtype, layout", DTYPE_LAYOUTS)
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES)
def test_all_gather_gather_dim_0(mesh_device, dtype, layout, shard_shape):
    """Every device's output equals the concat of all shards along gather_dim=0."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("all_gather requires at least 2 mesh devices")

    input_tensor, torch_full = _make_sharded_input(mesh_device, shard_shape, dtype, layout)

    output_tensor = all_gather(input_tensor, 0, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]

    pcc = PCC[dtype]
    torch_ref = torch_full.to(torch.float32)
    # Every device now holds the full concatenated tensor.
    for dev_idx, dev_out in enumerate(output_shards):
        assert tuple(dev_out.shape) == tuple(
            torch_full.shape
        ), f"device {dev_idx} output shape {tuple(dev_out.shape)} != full {tuple(torch_full.shape)}"
        assert_with_pcc(torch_ref, dev_out.to(torch.float32), pcc)
    logger.info(
        f"all_gather {dtype} {layout} shard={shard_shape} on {num_devices} devices: every device holds the full tensor"
    )


@pytest.mark.parametrize("device_params", [LINEAR], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_all_gather_program_cache(mesh_device):
    """Second call (program-cache hit) still gathers correctly.

    The op-internal GlobalSemaphore(s) must survive the cache hit (created once,
    not re-created per call).
    """
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("all_gather requires at least 2 mesh devices")

    for call in range(2):
        input_tensor, torch_full = _make_sharded_input(mesh_device, (1, 1, 32, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        output_tensor = all_gather(input_tensor, 0, topology=ttnn.Topology.Linear)
        ttnn.synchronize_device(mesh_device)
        output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
        for dev_out in output_shards:
            assert_with_pcc(torch_full.to(torch.float32), dev_out.to(torch.float32), PCC[ttnn.bfloat16])
        logger.info(f"program-cache call {call}: all {num_devices} devices hold the full tensor")

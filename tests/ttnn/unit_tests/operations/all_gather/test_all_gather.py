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

This file is the immutable spec — the implementer must not modify it.

Verification topology (MUST match the sim's fixed mesh-graph descriptor):
a Wormhole T3K **line mesh of shape (1, 8)** with
``fabric_config = ttnn.FabricConfig.FABRIC_1D``, driven by
``scripts/run_multidevice_sim_pytest.py --op all_gather``. A different mesh
shape hangs fabric init ("Fabric Router Sync: Timeout"). The proven first case
is ``gather_dim=0``, bfloat16, TILE_LAYOUT, Linear topology.
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

# Valid (dtype, layout) pairs. bfloat8_b is a tiled block-float format with no
# row-major representation, so it appears only with TILE_LAYOUT.
DTYPE_LAYOUTS = [
    (ttnn.bfloat16, ttnn.TILE_LAYOUT),
    (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    (ttnn.float32, ttnn.TILE_LAYOUT),
    (ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
]

# Per-device shard shapes: single-tile, multi-tile, non-square, multi-batch,
# non-tile-aligned. Last dims are multiples of 8 so the row-major page size
# stays 16-byte aligned for every dtype.
SHARD_SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 64, 128),  # multi-tile
    (1, 1, 96, 64),  # non-square, tile-aligned
    (2, 1, 32, 64),  # multi-batch
    (1, 1, 48, 64),  # non-tile-aligned (H not %32), 16B-aligned page
]

# Topology <-> fabric_config pairing. The sim is a FABRIC_1D line.
LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)


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


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("dtype, layout", DTYPE_LAYOUTS)
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES)
def test_all_gather_gather_dim_0(mesh_device, topology, dtype, layout, shard_shape):
    """Every device's output equals the concat of all shards along gather_dim=0."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("all_gather requires at least 2 mesh devices")

    input_tensor, torch_full = _make_sharded_input(mesh_device, shard_shape, dtype, layout)

    output_tensor = all_gather(input_tensor, 0, topology=topology)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]

    pcc = PCC[dtype]
    # Every device now holds the full concatenated tensor.
    for dev_idx, dev_out in enumerate(output_shards):
        assert tuple(dev_out.shape) == tuple(
            torch_full.shape
        ), f"device {dev_idx} output shape {tuple(dev_out.shape)} != full {tuple(torch_full.shape)}"
        assert_with_pcc(torch_full, dev_out, pcc)
    logger.info(
        f"all_gather {dtype} {layout} shard={shard_shape} {topology}: all {num_devices} devices hold the full tensor"
    )


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_all_gather_program_cache(mesh_device, topology):
    """Second call (program-cache hit) still gathers correctly.

    The op-internal GlobalSemaphore must survive the cache hit (created once,
    not re-created per call).
    """
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("all_gather requires at least 2 mesh devices")

    for call in range(2):
        input_tensor, torch_full = _make_sharded_input(mesh_device, (1, 1, 32, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        output_tensor = all_gather(input_tensor, 0, topology=topology)
        ttnn.synchronize_device(mesh_device)
        output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
        for dev_out in output_shards:
            assert_with_pcc(torch_full, dev_out, PCC[ttnn.bfloat16])
        logger.info(f"program-cache call {call}: all {num_devices} devices hold the full tensor")


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_all_gather_output_tensor(mesh_device, topology):
    """The output_tensor path writes into the supplied tensor and returns it."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("all_gather requires at least 2 mesh devices")

    shard_shape = (1, 1, 64, 128)
    input_tensor, torch_full = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT)

    # Pre-allocate a full-gathered-shape output buffer on every device (replicated
    # zeros; all_gather overwrites every page). This sidesteps manual TensorSpec
    # construction and yields a properly-allocated per-device output handle.
    out_shape = (shard_shape[0] * num_devices, *shard_shape[1:])
    preallocated = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.synchronize_device(mesh_device)

    returned = all_gather(input_tensor, 0, topology=topology, output_tensor=preallocated)
    ttnn.synchronize_device(mesh_device)

    # Same handle is returned.
    assert returned.buffer_address() == preallocated.buffer_address()

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(returned)]
    for dev_out in output_shards:
        assert_with_pcc(torch_full, dev_out, PCC[ttnn.bfloat16])

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for the self-contained Python all_reduce CCL op (CCL + compute).

all_reduce sums each device's shard element-wise across all N devices on a
MeshDevice line and leaves the IDENTICAL sum on every device. Unlike the
pure-movement CCLs (point_to_point identity, all_gather concat), the element
values change (they are summed), so this op combines the fabric dataflow
movement with a compute (TRISC) reduction. The oracle is element-wise SUM:

  * each device's output == the host-side element-wise sum of all N devices'
    input shards (same shape/dtype/layout as a single input shard).

This file is the immutable spec — the implementer must not modify it.

Verification topology (MUST match the sim's fixed mesh-graph descriptor):
a Wormhole T3K **line mesh of shape (1, 8)** with
``fabric_config = ttnn.FabricConfig.FABRIC_1D``, driven by
``scripts/run_multidevice_sim_pytest.py --op all_reduce``. A different mesh
shape hangs fabric init ("Fabric Router Sync: Timeout"). The proven first case
is bfloat16, TILE_LAYOUT, Linear topology.
"""

from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.all_reduce import all_reduce


# PCC tolerances keyed by dtype. A bf16 sum of N terms accumulates rounding, so
# the bf16 threshold matches the all_reduce golden suite (0.99) rather than the
# generic pure-movement 0.995 — the reduction genuinely loses a little precision.
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.99,
}

# Reduction runs on TILE_LAYOUT (it is a tile compute). bf16 is the proven
# primary dtype; float32 is the secondary supported dtype.
DTYPES = [ttnn.bfloat16, ttnn.float32]

# Per-device shard shapes: single-tile, multi-tile, non-square, multi-batch.
# Every device holds a shard of the SAME shape (distinct values); all_reduce
# sums the N shards element-wise. All tile-aligned (last two dims % 32 == 0).
SHARD_SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 64, 128),  # multi-tile
    (1, 1, 128, 64),  # non-square, tile-aligned
    (2, 1, 32, 64),  # multi-batch
]

# Topology <-> fabric_config pairing. The sim is a FABRIC_1D line.
LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)


def _make_sharded_input(mesh_device, shard_shape, dtype):
    """Shard a freshly-seeded full tensor along dim 0 across the whole line.

    The full tensor has shape ``(N * shard_shape[0], *shard_shape[1:])``; each
    device receives exactly ``shard_shape`` (distinct values). Returns the ttnn
    input tensor and the torch SUM oracle (element-wise sum of the N shards,
    accumulated in fp32 then cast, so the reference is not itself limited by
    bf16 rounding).
    """
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])

    torch.manual_seed(42)
    torch_full = torch.randn(full_shape, dtype=torch.float32)

    oracle = torch_full.reshape(num_devices, *shard_shape).sum(dim=0)
    if dtype == ttnn.bfloat16:
        torch_full = torch_full.to(torch.bfloat16)
        oracle = oracle.to(torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    return input_tensor, oracle


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES)
def test_all_reduce(mesh_device, topology, dtype, shard_shape):
    """Every device's output equals the element-wise SUM of all N shards."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("all_reduce requires at least 2 mesh devices")

    input_tensor, oracle = _make_sharded_input(mesh_device, shard_shape, dtype)

    output_tensor = all_reduce(input_tensor, topology=topology)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]

    pcc = PCC[dtype]
    # Every device now holds the identical element-wise sum of all N shards.
    for dev_idx, dev_out in enumerate(output_shards):
        assert tuple(dev_out.shape) == tuple(
            shard_shape
        ), f"device {dev_idx} output shape {tuple(dev_out.shape)} != shard {tuple(shard_shape)}"
        assert_with_pcc(oracle, dev_out, pcc)
    logger.info(
        f"all_reduce {dtype} shard={shard_shape} {topology}: all {num_devices} devices hold the element-wise sum"
    )


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_all_reduce_program_cache(mesh_device, topology):
    """Second call (program-cache hit) still reduces correctly.

    The op-internal GlobalSemaphore must survive the cache hit (created once,
    not re-created per call).
    """
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("all_reduce requires at least 2 mesh devices")

    for call in range(2):
        input_tensor, oracle = _make_sharded_input(mesh_device, (1, 1, 32, 64), ttnn.bfloat16)
        output_tensor = all_reduce(input_tensor, topology=topology)
        ttnn.synchronize_device(mesh_device)
        output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
        for dev_out in output_shards:
            assert_with_pcc(oracle, dev_out, PCC[ttnn.bfloat16])
        logger.info(f"program-cache call {call}: all {num_devices} devices hold the element-wise sum")


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_all_reduce_output_tensor(mesh_device, topology):
    """The output_tensor path writes into the supplied tensor and returns it."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("all_reduce requires at least 2 mesh devices")

    shard_shape = (1, 1, 64, 128)
    input_tensor, oracle = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16)

    # Pre-allocate a shard-shape output buffer on every device (replicated
    # zeros; all_reduce overwrites every page). Yields a properly-allocated
    # per-device output handle without manual TensorSpec construction.
    preallocated = ttnn.from_torch(
        torch.zeros(shard_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.synchronize_device(mesh_device)

    returned = all_reduce(input_tensor, topology=topology, output_tensor=preallocated)
    ttnn.synchronize_device(mesh_device)

    # Same handle is returned.
    assert returned.buffer_address() == preallocated.buffer_address()

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(returned)]
    for dev_out in output_shards:
        assert_with_pcc(oracle, dev_out, PCC[ttnn.bfloat16])

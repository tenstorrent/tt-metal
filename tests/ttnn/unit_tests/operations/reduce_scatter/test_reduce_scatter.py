# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for the self-contained Python reduce_scatter CCL op (the compute-CCL probe).

reduce_scatter sums each device's shard element-wise across all N devices on a
MeshDevice line, then SCATTERS the sum: device i's output is the i-th of N equal
slices of the summed tensor along ``dim`` (Phase-0: dim=3, the last dim). Unlike
all_reduce (identical sum everywhere), the output is per-device DISTINCT, so the
oracle is sum-then-slice:

  * device i's output == slice i (of N chunks along dim) of the host-side
    element-wise sum of all N devices' input shards, with
    ``output.shape[dim] == shard.shape[dim] / N``.

This file is the immutable spec — the implementer must not modify it.

Verification topology: ``bh_quietbox_1x4_hw`` — a 4-chip Blackhole mesh of shape
**(1, 4)** with ``fabric_config = ttnn.FabricConfig.FABRIC_1D``, on REAL HARDWARE
via ``scripts/run_multidevice_sim_pytest.py --runtime hardware --op reduce_scatter``.
A different mesh shape hangs fabric init ("Fabric Router Sync: Timeout") or fails
``system_mesh.cpp: requested_size <= system_size``. The runner exports the active
topology's shape as ``MULTIDEV_SIM_MESH_SHAPE``; this file reads it with the
hardware contract's (1, 4) as the default, so the same spec also runs unmodified
on the (1, 8) sim line. All shard widths are multiples of 256 = 32 * 8 so the
per-device output slice (width / N) stays tile-aligned for N in {4, 8}.
"""

import os
from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.reduce_scatter import SUPPORTED, reduce_scatter


# The verification topology's mesh shape. (1, 4) is the bh_quietbox_1x4_hw
# hardware contract; the multidevice runner overrides via MULTIDEV_SIM_MESH_SHAPE
# for its other (sim) topologies. Do NOT hardcode a different shape.
MESH_SHAPE = tuple(int(x) for x in os.environ.get("MULTIDEV_SIM_MESH_SHAPE", "1,4").split(","))

# PCC tolerances keyed by dtype. A bf16 sum of N terms accumulates rounding, so
# the bf16 threshold matches the reduce_scatter golden suite (0.99), not the
# generic pure-movement 0.995 — the reduction genuinely loses a little precision.
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.99,
}

# The reduction runs on TILE_LAYOUT (it is a tile compute). bf16 is the proven
# primary dtype; float32 is the secondary supported dtype.
DTYPES = [ttnn.bfloat16, ttnn.float32]

# Per-device shard shapes: smallest, multi-tile, wide non-square, multi-batch.
# Every device holds a shard of the SAME shape (distinct values). Widths are
# multiples of 256 so width / N is tile-aligned on both the (1, 4) hardware box
# and the (1, 8) sim line; all dims are tile-aligned (% 32 == 0).
SHARD_SHAPES = [
    (1, 1, 32, 256),  # smallest: one tile row; slice = 2 tiles (N=4) / 1 tile (N=8)
    (1, 1, 64, 512),  # multi-tile rows and columns
    (1, 1, 256, 256),  # tall square
    (2, 1, 64, 256),  # multi-batch
]

SCATTER_DIM = 3  # Phase-0 proven scatter dim (the last dim)

# Topology <-> fabric_config pairing. The verification fabric is a FABRIC_1D line.
LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)


def _make_sharded_input(mesh_device, shard_shape, dtype):
    """Shard a freshly-seeded full tensor along dim 0 across the whole line.

    The full tensor has shape ``(N * shard_shape[0], *shard_shape[1:])``; device i
    receives exactly ``shard_shape`` (distinct values). Returns the ttnn input
    tensor and the per-device torch oracle slices: the element-wise SUM of the N
    shards (accumulated in fp32 then cast, so the reference is not itself limited
    by bf16 rounding), chunked into N equal slices along SCATTER_DIM.
    """
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])

    torch.manual_seed(42)
    torch_full = torch.randn(full_shape, dtype=torch.float32)

    summed = torch_full.reshape(num_devices, *shard_shape).sum(dim=0)
    if dtype == ttnn.bfloat16:
        torch_full = torch_full.to(torch.bfloat16)
        summed = summed.to(torch.bfloat16)
    oracle_slices = torch.chunk(summed, num_devices, dim=SCATTER_DIM)

    input_tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    return input_tensor, oracle_slices


def _expected_output_shape(shard_shape, num_devices):
    out_shape = list(shard_shape)
    out_shape[SCATTER_DIM] //= num_devices
    return tuple(out_shape)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES)
def test_reduce_scatter(mesh_device, topology, dtype, shard_shape):
    """Device i's output equals slice i of the element-wise SUM of all N shards."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter requires at least 2 mesh devices")

    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, dtype)

    output_tensor = reduce_scatter(input_tensor, dim=SCATTER_DIM, topology=topology)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
    assert len(output_shards) == num_devices

    expected_shape = _expected_output_shape(shard_shape, num_devices)
    pcc = PCC[dtype]
    # Per-device DISTINCT outputs: device i holds slice i of the sum.
    for dev_idx, dev_out in enumerate(output_shards):
        assert (
            tuple(dev_out.shape) == expected_shape
        ), f"device {dev_idx} output shape {tuple(dev_out.shape)} != expected {expected_shape}"
        assert_with_pcc(oracle_slices[dev_idx], dev_out, pcc)
    logger.info(
        f"reduce_scatter {dtype} shard={shard_shape} dim={SCATTER_DIM} {topology}: "
        f"all {num_devices} devices hold their distinct slice of the sum"
    )


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_negative_dim_alias(mesh_device, topology):
    """dim=-1 is canonicalized to the positive convention and behaves as dim=3."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter requires at least 2 mesh devices")

    shard_shape = (1, 1, 32, 256)
    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16)

    output_tensor = reduce_scatter(input_tensor, dim=-1, topology=topology)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
    for dev_idx, dev_out in enumerate(output_shards):
        assert_with_pcc(oracle_slices[dev_idx], dev_out, PCC[ttnn.bfloat16])


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_program_cache(mesh_device, topology):
    """Second call (program-cache hit) still reduce-scatters correctly.

    The op-internal GlobalSemaphore must survive the cache hit (created once,
    not re-created per call), and the kernels must re-arm their counting
    semaphores (a missing ``noc_semaphore_set(sem, 0)`` re-arm passes the first
    call and hangs or corrupts the second — exactly what this test catches).
    """
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter requires at least 2 mesh devices")

    for call in range(2):
        input_tensor, oracle_slices = _make_sharded_input(mesh_device, (1, 1, 32, 256), ttnn.bfloat16)
        output_tensor = reduce_scatter(input_tensor, dim=SCATTER_DIM, topology=topology)
        ttnn.synchronize_device(mesh_device)
        output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
        for dev_idx, dev_out in enumerate(output_shards):
            assert_with_pcc(oracle_slices[dev_idx], dev_out, PCC[ttnn.bfloat16])
        logger.info(f"program-cache call {call}: all {num_devices} devices hold their slice of the sum")


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_output_tensor(mesh_device, topology):
    """The output_tensor path writes into the supplied tensor and returns it."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter requires at least 2 mesh devices")

    shard_shape = (1, 1, 64, 512)
    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16)

    # Pre-allocate an output-shape buffer on every device (replicated zeros; the
    # op overwrites every output page). Output slice shape = shard with dim/N.
    out_shape = _expected_output_shape(shard_shape, num_devices)
    preallocated = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.synchronize_device(mesh_device)

    returned = reduce_scatter(input_tensor, dim=SCATTER_DIM, topology=topology, output_tensor=preallocated)
    ttnn.synchronize_device(mesh_device)

    # Same handle is returned.
    assert returned.buffer_address() == preallocated.buffer_address()

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(returned)]
    for dev_idx, dev_out in enumerate(output_shards):
        assert tuple(dev_out.shape) == out_shape
        assert_with_pcc(oracle_slices[dev_idx], dev_out, PCC[ttnn.bfloat16])


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_rejects_indivisible_shape(mesh_device, topology):
    """A scatter dim not divisible into tile-aligned slices raises ValueError (no silent pad).

    96 = 3 tiles: tile-aligned as a shard, but 96 % (N * 32) != 0 for N in
    {4, 8}, so the per-device output slice cannot be a whole number of tiles.
    """
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter requires at least 2 mesh devices")

    input_tensor, _ = _make_sharded_input(mesh_device, (1, 1, 32, 96), ttnn.bfloat16)
    with pytest.raises(ValueError):
        reduce_scatter(input_tensor, dim=SCATTER_DIM, topology=topology)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_unsupported_dim_refuses(mesh_device, topology):
    """An out-of-SUPPORTED scatter dim raises NotImplementedError (UnsupportedAxisValue).

    dim=2 is a TARGET refinement axis value; while it is outside SUPPORTED the
    registry contract requires the typed refusal (a NotImplementedError
    subclass), not a hard failure. The shape is structurally valid for dim=2
    (256 % (N * 32) == 0 for N in {4, 8}), so only the axis gate can refuse.
    """
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter requires at least 2 mesh devices")
    if 2 in SUPPORTED.get("dim", []):
        pytest.skip("dim=2 has been refined into SUPPORTED; the refusal contract no longer applies")

    input_tensor, _ = _make_sharded_input(mesh_device, (1, 1, 256, 256), ttnn.bfloat16)
    with pytest.raises(NotImplementedError):
        reduce_scatter(input_tensor, dim=2, topology=topology)

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for the self-contained Python reduce_scatter CCL op (CCL + compute).

reduce_scatter sums each device's shard element-wise across all N devices on a
MeshDevice line, then SCATTERS the sum: device i's output is the i-th of N equal
slices of the summed tensor along ``dim`` (Phase-0: dim=3, the last dim). The
oracle is SUM-THEN-SLICE — per-device DISTINCT outputs, unlike all_reduce's
identical-everywhere sum, so a schedule/addressing bug (wrong slice reduced)
is visible here where all_reduce's oracle can mask it:

  * device i's output == slice i (of N equal slices along dim 3) of the
    host-side element-wise sum of all N devices' input shards
    (shape = shard shape with shape[3] / N; same dtype/layout).

This file is the immutable spec — the implementer must not modify it.

Verification topology (the contract — MUST match this fixture): REAL BLACKHOLE
HARDWARE, a **4-chip mesh of shape (1, 4)** with
``fabric_config = ttnn.FabricConfig.FABRIC_1D``, driven by
``scripts/run_multidevice_sim_pytest.py --runtime hardware --op reduce_scatter``
(topology entry ``bh_quietbox_1x4_hw``). A different mesh shape hangs fabric
init ("Fabric Router Sync: Timeout") or fails
``system_mesh.cpp: requested_size <= system_size`` — either is a test/topology
mismatch, not an op defect. The proven first case is bfloat16, TILE_LAYOUT,
dim=3, Linear topology.
"""

from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.reduce_scatter import reduce_scatter


# PCC tolerances keyed by dtype. A bf16 sum of N terms accumulates rounding, so
# the bf16 threshold matches the reduce_scatter golden suite (0.99) rather than
# the generic pure-movement 0.995 — the reduction genuinely loses a little
# precision.
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.99,
}

# The reduction runs on TILE_LAYOUT (it is a tile compute). bf16 is the proven
# primary dtype; float32 is the secondary supported dtype.
DTYPES = [ttnn.bfloat16, ttnn.float32]

_SCATTER_DIM = 3  # Phase-0: the last dim

# Per-device shard shapes: single-output-tile, multi-tile, tall non-square,
# multi-batch. Every device holds a shard of the SAME shape (distinct values).
# shape[3] is a multiple of num_devices * 32 = 128 on the (1, 4) contract mesh,
# so each device's output slice is a whole number of tiles.
SHARD_SHAPES = [
    (1, 1, 32, 128),  # one output tile per device
    (1, 1, 64, 256),  # multi-tile rows and columns
    (1, 1, 128, 128),  # tall non-square (output 128 x 32)
    (2, 1, 32, 256),  # multi-batch
]

# Topology <-> fabric_config pairing. The verification mesh is a FABRIC_1D line.
LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)

# The verification mesh shape — the bh_quietbox_1x4_hw contract. Do NOT change.
MESH_SHAPE = (1, 4)


def _make_sharded_input(mesh_device, shard_shape, dtype):
    """Shard a freshly-seeded full tensor along dim 0 across the whole line.

    The full tensor has shape ``(N * shard_shape[0], *shard_shape[1:])``; device
    i receives exactly ``full.reshape(N, *shard_shape)[i]`` (distinct values).
    Returns the ttnn input tensor and the per-device torch oracles: slice i (of
    N equal slices along dim 3) of the element-wise sum of the N shards. The
    sum is accumulated in fp32 then cast, so the reference is not itself
    limited by bf16 rounding.
    """
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])

    torch.manual_seed(42)
    torch_full = torch.randn(full_shape, dtype=torch.float32)

    summed = torch_full.reshape(num_devices, *shard_shape).sum(dim=0)
    if dtype == ttnn.bfloat16:
        torch_full = torch_full.to(torch.bfloat16)
        summed = summed.to(torch.bfloat16)
    oracle_slices = torch.chunk(summed, num_devices, dim=_SCATTER_DIM)

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


def _expected_out_shape(shard_shape, num_devices):
    out_shape = list(shard_shape)
    out_shape[_SCATTER_DIM] //= num_devices
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

    output_tensor = reduce_scatter(input_tensor, dim=_SCATTER_DIM, topology=topology)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]

    pcc = PCC[dtype]
    expected_shape = _expected_out_shape(shard_shape, num_devices)
    # Per-device DISTINCT outputs: device i holds slice i of the sum.
    for dev_idx, dev_out in enumerate(output_shards):
        assert (
            tuple(dev_out.shape) == expected_shape
        ), f"device {dev_idx} output shape {tuple(dev_out.shape)} != expected {expected_shape}"
        assert_with_pcc(oracle_slices[dev_idx], dev_out, pcc)
    logger.info(
        f"reduce_scatter {dtype} shard={shard_shape} {topology}: "
        f"each of {num_devices} devices holds its own slice of the sum"
    )


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_negative_dim_alias(mesh_device, topology):
    """dim=-1 is the same axis as dim=3 (canonicalized before the support gate)."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter requires at least 2 mesh devices")

    shard_shape = (1, 1, 32, 128)
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
    not re-created per call), and the Phase-A readers must have re-armed it
    (``noc_semaphore_set(sem, 0)``) — a missing re-arm passes call 1 and hangs
    call 2. The cache-entry count must not grow on the second call.
    """
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter requires at least 2 mesh devices")

    shard_shape = (1, 1, 32, 128)
    entries_after_first = None
    for call in range(2):
        input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16)
        output_tensor = reduce_scatter(input_tensor, dim=_SCATTER_DIM, topology=topology)
        ttnn.synchronize_device(mesh_device)
        output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
        for dev_idx, dev_out in enumerate(output_shards):
            assert_with_pcc(oracle_slices[dev_idx], dev_out, PCC[ttnn.bfloat16])
        if call == 0:
            entries_after_first = mesh_device.num_program_cache_entries()
        else:
            assert mesh_device.num_program_cache_entries() == entries_after_first, (
                "second same-shape call must hit the program cache "
                f"(entries grew from {entries_after_first} to "
                f"{mesh_device.num_program_cache_entries()})"
            )
        logger.info(f"program-cache call {call}: every device holds its own slice of the sum")


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_output_tensor(mesh_device, topology):
    """The output_tensor path writes into the supplied tensor and returns it."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter requires at least 2 mesh devices")

    shard_shape = (1, 1, 64, 256)
    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16)

    # Pre-allocate an output-slice-shape buffer on every device (replicated
    # zeros; reduce_scatter overwrites every page). Yields a properly-allocated
    # per-device output handle without manual TensorSpec construction.
    out_shape = _expected_out_shape(shard_shape, num_devices)
    preallocated = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.synchronize_device(mesh_device)

    returned = reduce_scatter(input_tensor, dim=_SCATTER_DIM, topology=topology, output_tensor=preallocated)
    ttnn.synchronize_device(mesh_device)

    # Same handle is returned.
    assert returned.buffer_address() == preallocated.buffer_address()

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(returned)]
    for dev_idx, dev_out in enumerate(output_shards):
        assert_with_pcc(oracle_slices[dev_idx], dev_out, PCC[ttnn.bfloat16])


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_rejects_unsliceable_width(mesh_device, topology):
    """shape[3] not divisible by N * 32 is rejected loudly (ValueError, no pad).

    W = 64 on a 4-device line would need a 16-element (half-tile) output slice.
    """
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter requires at least 2 mesh devices")

    shard_shape = (1, 1, 32, 64)  # 64 % (4 * 32) != 0
    input_tensor, _ = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16)

    with pytest.raises(ValueError):
        reduce_scatter(input_tensor, dim=_SCATTER_DIM, topology=topology)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_unsupported_dim_refuses(mesh_device, topology):
    """A dim outside SUPPORTED refuses with the registry-model NotImplementedError
    subclass (UnsupportedAxisValue) — the refinement-candidate contract."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter requires at least 2 mesh devices")

    shard_shape = (1, 1, 32, 128)
    input_tensor, _ = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16)

    with pytest.raises(NotImplementedError):
        reduce_scatter(input_tensor, dim=2, topology=topology)

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for reduce_scatter_average (IMMUTABLE — the implementer must not modify it).

reduce_scatter_average = reduce_scatter with a SUM reduction scaled by 1/N before
landing: device i's output is slice i (of N equal slices along ``dim``) of the
element-wise MEAN of all N devices' shards. The scaling is part of the op — the
caller passes nothing but the tensor.

Verification topology (MUST match the sim's fixed mesh-graph descriptor):
``wh_t3k_allmmio_reduce_scatter_average`` — a wormhole_b0 mesh of shape (1, 8)
with ``fabric_config = ttnn.FabricConfig.FABRIC_1D``. A different mesh shape
hangs fabric init ("Fabric Router Sync: Timeout") or fails
``system_mesh.cpp: requested_size <= system_size``.

Run via the multichip sim runner, NEVER run_safe_pytest.sh:
    scripts/run_multidevice_sim_pytest.py --op reduce_scatter_average \
        tests/ttnn/unit_tests/operations/reduce_scatter_average/test_reduce_scatter_average.py
"""

from math import prod

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.reduce_scatter_average import reduce_scatter_average

# Reductions get a looser bf16 tolerance than pure-movement CCLs: a bf16 chain of
# N partial sums accumulates rounding at every hop even though DEST accumulates in
# fp32 (partials are stored bf16 in the intermediate buffer). Matches the
# all_reduce / reduce_scatter golden-suite thresholds — do not tighten.
PCC = {
    ttnn.bfloat16: 0.99,
    ttnn.float32: 0.999,
}

DTYPES = [ttnn.bfloat16, ttnn.float32]
TORCH_DTYPES = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}

# Per-device shard shapes (B, C, H, W). W is a multiple of 256 = 8 devices x 32
# so each device's output slice (W / N) stays tile-aligned on the (1, 8) grade
# mesh (and on a CCL_HW_MESH_SHAPE=1,4 hardware box).
SHARD_SHAPES = [
    (1, 1, 32, 256),  # minimal H: one tile row, one output tile per device on (1, 8)
    (1, 1, 256, 256),  # square, multi-tile rows and columns
    (1, 1, 64, 512),  # non-square, wider than tall
    (2, 1, 96, 256),  # batch > 1: exercises the schedule's per-batch restart
]

# The fabric config goes through the device_params fixture (indirect) while
# topology stays a plain value — pairing them in one parametrize keeps the two
# provably consistent.
LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)


def _mesh_inputs(mesh_device, shard_shape, dtype):
    """Build the sharded mesh input and the per-device fp32 mean-then-slice oracle."""
    num_devices = prod(tuple(mesh_device.shape))

    torch.manual_seed(42)
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])
    torch_full = torch.randn(full_shape, dtype=torch.float32)

    # Oracle accumulates in fp32 THEN casts, so the reference is not itself
    # limited by bf16 rounding: mean over shards, then N equal slices on dim 3.
    mean = torch_full.reshape(num_devices, *shard_shape).sum(dim=0) / num_devices
    oracle_slices = torch.chunk(mean.to(TORCH_DTYPES[dtype]), num_devices, dim=3)

    if dtype == ttnn.bfloat16:
        torch_full = torch_full.to(torch.bfloat16)

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


def _check_outputs(mesh_device, output_tensor, oracle_slices, shard_shape, dtype):
    num_devices = prod(tuple(mesh_device.shape))
    expected_shape = list(shard_shape)
    expected_shape[3] //= num_devices

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
    assert len(output_shards) == num_devices
    for dev_idx, dev_out in enumerate(output_shards):
        assert (
            list(dev_out.shape) == expected_shape
        ), f"device {dev_idx}: output shape {list(dev_out.shape)} != expected {expected_shape}"
        assert_with_pcc(oracle_slices[dev_idx], dev_out, PCC[dtype])


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES)
def test_reduce_scatter_average(mesh_device, topology, dtype, shard_shape):
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter_average requires at least 2 mesh devices")

    input_tensor, oracle_slices = _mesh_inputs(mesh_device, shard_shape, dtype)

    output_tensor = reduce_scatter_average(input_tensor, dim=3, topology=topology)
    ttnn.synchronize_device(mesh_device)

    _check_outputs(mesh_device, output_tensor, oracle_slices, shard_shape, dtype)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_reduce_scatter_average_negative_dim_alias(mesh_device, topology):
    """dim=-1 is the same axis as dim=3 and must be accepted (positive canonicalization)."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter_average requires at least 2 mesh devices")

    shard_shape = (1, 1, 32, 256)
    dtype = ttnn.bfloat16
    input_tensor, oracle_slices = _mesh_inputs(mesh_device, shard_shape, dtype)

    output_tensor = reduce_scatter_average(input_tensor, dim=-1, topology=topology)
    ttnn.synchronize_device(mesh_device)

    _check_outputs(mesh_device, output_tensor, oracle_slices, shard_shape, dtype)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_reduce_scatter_average_program_cache(mesh_device, topology):
    """Second identical call must hit the program cache and still be correct — the
    op-internal GlobalSemaphores must survive the cache hit (created once per mesh,
    parked on the mesh program descriptor, kernel-side re-arm resets)."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter_average requires at least 2 mesh devices")

    shard_shape = (1, 1, 256, 256)
    dtype = ttnn.bfloat16
    input_tensor, oracle_slices = _mesh_inputs(mesh_device, shard_shape, dtype)

    # Call 2 is the trap: a missing kernel-side semaphore re-arm or a per-call
    # semaphore re-creation is "green run 1, hang or corrupt run 2".
    for call in range(2):
        out = reduce_scatter_average(input_tensor, dim=3, topology=topology)
        ttnn.synchronize_device(mesh_device)
        _check_outputs(mesh_device, out, oracle_slices, shard_shape, dtype)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_reduce_scatter_average_output_tensor(mesh_device, topology):
    """The output_tensor path writes into the supplied tensor and returns it."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter_average requires at least 2 mesh devices")

    shard_shape = (1, 1, 64, 256)
    dtype = ttnn.bfloat16
    input_tensor, oracle_slices = _mesh_inputs(mesh_device, shard_shape, dtype)

    out_shape = list(shard_shape)
    out_shape[3] //= num_devices
    preallocated = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.bfloat16),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.synchronize_device(mesh_device)

    returned = reduce_scatter_average(input_tensor, dim=3, topology=topology, output_tensor=preallocated)
    ttnn.synchronize_device(mesh_device)

    assert (
        returned.buffer_address() == preallocated.buffer_address()
    ), "output_tensor path must write into (and return) the supplied tensor"
    _check_outputs(mesh_device, returned, oracle_slices, shard_shape, dtype)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_reduce_scatter_average_rejects_unaligned_slice(mesh_device, topology):
    """shape[dim] not divisible by N*32 must raise ValueError loudly — never pad."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter_average requires at least 2 mesh devices")

    # W = 32*N would be fine; W = 32 (one tile total) is not divisible into N
    # tile-aligned slices for N >= 2.
    shard_shape = (1, 1, 32, 32)
    torch.manual_seed(42)
    torch_full = torch.randn((num_devices, 1, 32, 32), dtype=torch.float32).to(torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_full,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)

    with pytest.raises(ValueError):
        reduce_scatter_average(input_tensor, dim=3, topology=topology)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_reduce_scatter_average_unsupported_axis_refuses_typed(mesh_device, topology):
    """Out-of-SUPPORTED axis values must raise the typed refusal (a
    NotImplementedError subclass), not a generic error — the registry contract."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter_average requires at least 2 mesh devices")

    input_tensor, _ = _mesh_inputs(mesh_device, (1, 1, 256, 256), ttnn.bfloat16)

    # dim=2 is in TARGET but outside Phase-0 SUPPORTED.
    with pytest.raises(NotImplementedError):
        reduce_scatter_average(input_tensor, dim=2, topology=topology)

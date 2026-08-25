# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Acceptance tests for the generated reduce_scatter CCL op.

reduce_scatter sums each device's same-shape shard element-wise across all N
devices on a MeshDevice line, then scatters the sum: device i's output is the
i-th of N equal slices along `dim` (per-device DISTINCT, unlike all_reduce).

Verification topology contract (bh_quietbox_1x4_hw): a 4-chip Blackhole mesh of
shape (1, 4) with fabric_config = ttnn.FabricConfig.FABRIC_1D. The mesh_device
fixture below MUST open exactly (1, 4) — any other shape hangs fabric init
("Fabric Router Sync: Timeout") or fails system_mesh.cpp's size check; either is
a test/topology mismatch, not an op defect. Drive on hardware via
scripts/run_multidevice_sim_pytest.py --runtime hardware --op reduce_scatter
(NOT run_safe_pytest.sh, which is single-device oriented).

This file is the immutable acceptance spec — the implementer must not modify it.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.reduce_scatter import reduce_scatter

# PCC keyed by dtype: a bf16 sum of N terms accumulates rounding, so bf16 uses
# the reduction-CCL threshold 0.99 (same as the golden suite), fp32 0.999.
PCC = {ttnn.bfloat16: 0.99, ttnn.float32: 0.999}
TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}

# Topology <-> fabric_config pairing: parametrized together (indirect only on
# device_params) so the fabric config and the op kwarg can never drift.
LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)

# The verification mesh — the contract, do not change (see module docstring).
MESH_SHAPE = (1, 4)

# Per-device shard shapes. W must be a multiple of N*32 = 128 so every device's
# output slice (W / N) is tile-aligned. Coverage: single tile-row slice,
# multi-tile, non-square, multi-batch, and an odd tile count (S = 9) that forces
# the g = 1 granule path.
SHARD_SHAPES = [
    (1, 1, 32, 128),  # minimal: slice is a single tile (S = 1)
    (1, 1, 64, 256),  # multi-tile rows and columns (S = 4)
    (1, 1, 128, 128),  # non-square, tall (S = 4)
    (2, 1, 64, 256),  # multi-batch (S = 8)
    (1, 1, 96, 384),  # odd slice tile count (S = 9 -> granule g = 1)
]


def _num_devices(mesh_device):
    rows, cols = tuple(mesh_device.shape)
    return rows * cols


def _make_sharded_input(mesh_device, shard_shape, dtype, dim=3):
    """Build N distinct same-shape shards, shard them across the line mesh, and
    return (input_tensor, oracle_slices): oracle_slices[i] is the expected
    output of device i — slice i of the fp32-accumulated element-wise sum of
    the N quantized shards, cast back to the op dtype."""
    num_devices = _num_devices(mesh_device)
    torch_dtype = TORCH_DTYPE[dtype]
    torch.manual_seed(42)
    full = torch.randn((num_devices * shard_shape[0], *shard_shape[1:]), dtype=torch.float32)
    quantized = full.to(torch_dtype)

    # Oracle: sum the N shards the device actually sees (quantized), accumulated
    # in fp32 so the reference is not itself limited by bf16 rounding, then N
    # equal slices along `dim`; device i's expected output is slice i.
    summed = quantized.reshape(num_devices, *shard_shape).to(torch.float32).sum(dim=0).to(torch_dtype)
    oracle_slices = list(torch.chunk(summed, num_devices, dim=dim))

    input_tensor = ttnn.from_torch(
        quantized,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    return input_tensor, oracle_slices


def _check_outputs(mesh_device, output_tensor, oracle_slices, shard_shape, dtype, dim=3):
    num_devices = _num_devices(mesh_device)
    expected_shape = list(shard_shape)
    expected_shape[dim] //= num_devices
    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
    assert len(output_shards) == num_devices
    for i, dev_out in enumerate(output_shards):
        assert tuple(dev_out.shape) == tuple(
            expected_shape
        ), f"device {i}: output shape {tuple(dev_out.shape)} != expected {tuple(expected_shape)}"
        assert_with_pcc(oracle_slices[i], dev_out, PCC[dtype])


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES)
def test_reduce_scatter(mesh_device, topology, dtype, shard_shape):
    """Device i's output equals slice i of the element-wise sum of all shards."""
    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, dtype)
    output_tensor = reduce_scatter(input_tensor, dim=3, topology=topology)
    ttnn.synchronize_device(mesh_device)
    _check_outputs(mesh_device, output_tensor, oracle_slices, shard_shape, dtype)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_negative_dim_alias(mesh_device, topology):
    """dim=-1 canonicalizes to dim=3 (positive convention) and reduces correctly."""
    shard_shape = (1, 1, 64, 256)
    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16)
    output_tensor = reduce_scatter(input_tensor, dim=-1, topology=topology)
    ttnn.synchronize_device(mesh_device)
    _check_outputs(mesh_device, output_tensor, oracle_slices, shard_shape, ttnn.bfloat16)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_program_cache(mesh_device, topology):
    """Second call (program-cache hit) still reduce-scatters correctly.

    The op-internal GlobalSemaphores must survive the cache hit (created once
    per mesh_device, not re-created per call), and every consumer must re-arm
    its semaphore counter to 0 after its final wait — a cached CCL program that
    fails to re-arm passes iteration 0 and hangs or corrupts on iteration 1.
    """
    shard_shape = (1, 1, 64, 256)
    for _ in range(2):
        input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16)
        output_tensor = reduce_scatter(input_tensor, dim=3, topology=topology)
        ttnn.synchronize_device(mesh_device)
        _check_outputs(mesh_device, output_tensor, oracle_slices, shard_shape, ttnn.bfloat16)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_output_tensor(mesh_device, topology):
    """The output_tensor path writes into the supplied tensor and returns the
    SAME handle (asserted via buffer_address equality)."""
    shard_shape = (1, 1, 64, 256)
    num_devices = _num_devices(mesh_device)
    out_shape = list(shard_shape)
    out_shape[3] //= num_devices

    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16)
    # ReplicateTensorToMesh yields a properly-allocated per-device output handle
    # of the (smaller) slice shape without manual TensorSpec construction; the
    # op overwrites every page, so the zero seed is irrelevant.
    preallocated = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.synchronize_device(mesh_device)

    returned = reduce_scatter(input_tensor, dim=3, topology=topology, output_tensor=preallocated)
    ttnn.synchronize_device(mesh_device)

    assert (
        returned.buffer_address() == preallocated.buffer_address()
    ), "output_tensor path must write into and return the supplied tensor handle"
    _check_outputs(mesh_device, returned, oracle_slices, shard_shape, ttnn.bfloat16)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_rejects_non_tile_aligned_slice(mesh_device, topology):
    """shape[dim] that splits into non-tile-aligned per-device slices is
    rejected loudly (ValueError), never silently padded: W=64 on N=4 devices
    gives 16-wide slices (not a multiple of 32)."""
    input_tensor, _ = _make_sharded_input(mesh_device, (1, 1, 32, 64), ttnn.bfloat16)
    with pytest.raises(ValueError):
        reduce_scatter(input_tensor, dim=3, topology=topology)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_rejects_unsupported_dim(mesh_device, topology):
    """An out-of-SUPPORTED scatter dim raises the typed refusal
    (UnsupportedAxisValue, a NotImplementedError subclass) — the contract the
    golden harness's xfail-strict refinement cells rely on."""
    input_tensor, _ = _make_sharded_input(mesh_device, (1, 1, 64, 256), ttnn.bfloat16)
    with pytest.raises(NotImplementedError):
        reduce_scatter(input_tensor, dim=1, topology=topology)

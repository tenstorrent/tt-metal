# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Mesh-shape-adaptive DEBUG mirror of the immutable acceptance test.

DO NOT DELETE — documents the hardware verification path. The acceptance file
(test_reduce_scatter_average.py) hardcodes the (1, 8) grade mesh, which SKIPS on a
4-chip box (the root mesh_device fixture skips when the request exceeds available
devices). This mirror sizes the mesh from CCL_HW_MESH_SHAPE (default (1, 4) — the
bh_quietbox_1x4_hw topology), exactly like the other CCL hardware probes
(tests/nightly/t3000/ccl/test_point_to_point.py), so the same op body is verified
on real silicon. Run via:

    scripts/run_multidevice_sim_pytest.py --runtime hardware \
        --op reduce_scatter_average -- <this file> -x

It also carries deterministic debug cases (all-ones, chip-index constants) whose
intermediate values are hand-calculable — the first stop for any numerical issue.
"""

import os
from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.reduce_scatter_average import reduce_scatter_average

MESH_SHAPE = tuple(int(x) for x in os.environ.get("CCL_HW_MESH_SHAPE", "1,4").split(","))

PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.99,
}

DTYPES = [ttnn.bfloat16, ttnn.float32]

# Same shard shapes as the acceptance file: widths are multiples of 256, so the
# per-device slice stays tile-aligned on both the (1, 8) grade mesh and a (1, 4) box.
SHARD_SHAPES = [
    (1, 1, 32, 256),  # S=1 per device at N=8 (g=1); S=2 at N=4 (g=2)
    (1, 1, 256, 256),
    (1, 1, 64, 512),
    (2, 1, 32, 256),
]

LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)


def _make_sharded_input(mesh_device, shard_shape, dtype, torch_full=None):
    """Shard a full tensor along dim 0; return (ttnn input, list of per-device oracles).

    Oracle: shards quantized to dtype first, mean accumulated in fp32, sliced on dim 3
    — identical construction to the acceptance file.
    """
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])

    if torch_full is None:
        torch.manual_seed(42)
        torch_full = torch.randn(full_shape, dtype=torch.float32)
    assert tuple(torch_full.shape) == full_shape
    if dtype == ttnn.bfloat16:
        torch_full = torch_full.to(torch.bfloat16)

    shards = torch_full.reshape(num_devices, *shard_shape).to(torch.float32)
    mean = shards.mean(dim=0)
    oracle_slices = list(mean.chunk(num_devices, dim=3))
    if dtype == ttnn.bfloat16:
        oracle_slices = [s.to(torch.bfloat16) for s in oracle_slices]

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


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_all_ones_single_tile(mesh_device, topology):
    """All-ones input: mean of N all-ones shards = 1.0 everywhere. Every intermediate
    is hand-calculable: accumulator after pass k = (k+1).0; scale = (N)*(1/N) = 1.0."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("requires at least 2 mesh devices")

    W = 32 * num_devices  # one output tile per device
    shard_shape = (1, 1, 32, W)
    full = torch.ones((num_devices, 1, 32, W), dtype=torch.float32)
    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16, full)

    output_tensor = reduce_scatter_average(input_tensor, topology=topology)
    ttnn.synchronize_device(mesh_device)

    for dev_idx, dev_out in enumerate(ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)):
        expected = torch.ones(1, 1, 32, 32, dtype=torch.bfloat16)
        assert torch.allclose(
            dev_out.float(), expected.float(), rtol=0.02, atol=0.02
        ), f"device {dev_idx}: max diff {(dev_out.float() - expected.float()).abs().max()}"


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_chip_index_constants(mesh_device, topology):
    """Shard c is the constant (c+1): mean = (1+2+..+N)/N = (N+1)/2 everywhere.
    Distinguishes 'wrong contribution set' (e.g. a block counted twice / a block
    missed) from broadcast/scale bugs — each wrong sum lands on a distinct constant."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("requires at least 2 mesh devices")

    W = 32 * num_devices
    shard_shape = (1, 1, 32, W)
    full = torch.cat([torch.full((1, 1, 32, W), float(c + 1)) for c in range(num_devices)], dim=0)
    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16, full)

    expected_val = (num_devices + 1) / 2.0
    output_tensor = reduce_scatter_average(input_tensor, topology=topology)
    ttnn.synchronize_device(mesh_device)

    for dev_idx, dev_out in enumerate(ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)):
        expected = torch.full((1, 1, 32, 32), expected_val, dtype=torch.bfloat16)
        assert torch.allclose(
            dev_out.float(), expected.float(), rtol=0.02, atol=0.02
        ), f"device {dev_idx}: got {dev_out.float().mean().item():.4f}, expected {expected_val}"


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES)
def test_reduce_scatter_average_hw(mesh_device, topology, dtype, shard_shape):
    """Acceptance body on the env-sized mesh: device i's output equals slice i of the
    fp32-accumulated mean of all N shards."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("requires at least 2 mesh devices")

    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, dtype)

    output_tensor = reduce_scatter_average(input_tensor, topology=topology)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
    assert len(output_shards) == num_devices

    expected_shape = list(shard_shape)
    expected_shape[3] //= num_devices
    for dev_idx, dev_out in enumerate(output_shards):
        assert tuple(dev_out.shape) == tuple(expected_shape)
        assert_with_pcc(oracle_slices[dev_idx], dev_out, PCC[dtype])
    logger.info(f"reduce_scatter_average {dtype} shard={shard_shape}: all {num_devices} devices OK")


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_program_cache_hw(mesh_device, topology):
    """Second call (program-cache hit) still averages correctly — catches a missing
    semaphore re-arm (R1) as a hang or corruption on the second call."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("requires at least 2 mesh devices")

    shard_shape = (1, 1, 32, 256)
    for call in range(2):
        input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16)
        output_tensor = reduce_scatter_average(input_tensor, topology=topology)
        ttnn.synchronize_device(mesh_device)
        for dev_idx, dev_out in enumerate(ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)):
            assert_with_pcc(oracle_slices[dev_idx], dev_out, PCC[ttnn.bfloat16])
        logger.info(f"program-cache call {call}: OK")


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_output_tensor_hw(mesh_device, topology):
    """The output_tensor path writes into the supplied tensor and returns it."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("requires at least 2 mesh devices")

    shard_shape = (1, 1, 64, 512)
    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16)

    out_shape = list(shard_shape)
    out_shape[3] //= num_devices
    preallocated = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.synchronize_device(mesh_device)

    returned = reduce_scatter_average(input_tensor, topology=topology, output_tensor=preallocated)
    ttnn.synchronize_device(mesh_device)

    assert returned.buffer_address() == preallocated.buffer_address()
    for dev_idx, dev_out in enumerate(ttnn.to_torch(t) for t in ttnn.get_device_tensors(returned)):
        assert_with_pcc(oracle_slices[dev_idx], dev_out, PCC[ttnn.bfloat16])


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_rejects_unsplittable_width_hw(mesh_device, topology):
    """Unsplittable slice widths raise ValueError loudly (no silent padding)."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("requires at least 2 mesh devices")
    if 96 % num_devices == 0 and (96 // num_devices) % 32 == 0:
        pytest.skip("width 96 is splittable on this mesh; rejection case does not apply")

    shard_shape = (1, 1, 32, 96)
    input_tensor, _ = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16)

    with pytest.raises(ValueError):
        reduce_scatter_average(input_tensor, topology=topology)

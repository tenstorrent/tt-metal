# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Extended verification tests for reduce_scatter — gaps the acceptance/golden/
precision suites do not cover (verifier-authored; matrix deliberately small).

Coverage added here:
  * L1 interleaved input (every other suite uses DRAM) — validate() accepts
    interleaved DRAM or L1 and the memory_config propagates to the output.
  * The S = 256 resident-accumulator budget BOUNDARY, both sides: exactly at the
    Phase-0 cliff (runs; the largest per-core CB footprint the op ever allocates)
    and one tile-column past it (loud ValueError, the documented refinement seam).
  * float32 output_tensor path (acceptance covers bf16 only).
  * Loud-rejection edges: out-of-range dim, mismatched output_tensor spec.
  * dim=2 with B > 1 (Refinement 2 Done-when): the per-batch plane restart of
    the reduce reader's dim=2 walk on the golden (2,1,256,256) shape.

Multi-device — drive via the multi-device runner (mesh MUST match the topology):

    scripts/run_multidevice_sim_pytest.py --op reduce_scatter -- \
        tests/ttnn/unit_tests/operations/reduce_scatter/test_reduce_scatter_extended.py -v
"""

from math import prod

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.reduce_scatter import reduce_scatter

PCC = {ttnn.bfloat16: 0.99, ttnn.float32: 0.999}
TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}

LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)
MESH_SHAPE = (1, 4)  # bh_quietbox_1x4_hw contract (see test_reduce_scatter.py)


def _num_devices(mesh_device):
    return prod(tuple(mesh_device.shape))


def _make_sharded_input(mesh_device, shard_shape, dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG, dim=3):
    num_devices = _num_devices(mesh_device)
    torch.manual_seed(7)
    full = torch.randn((num_devices * shard_shape[0], *shard_shape[1:]), dtype=torch.float32)
    quantized = full.to(TORCH_DTYPE[dtype])
    summed = quantized.reshape(num_devices, *shard_shape).to(torch.float32).sum(dim=0).to(TORCH_DTYPE[dtype])
    oracle_slices = list(torch.chunk(summed, num_devices, dim=dim))
    input_tensor = ttnn.from_torch(
        quantized,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory_config,
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
        assert tuple(dev_out.shape) == tuple(expected_shape)
        assert_with_pcc(oracle_slices[i], dev_out, PCC[dtype])


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_l1_interleaved(mesh_device, topology):
    """L1 interleaved input: validate() accepts it, the gather buffer and output
    inherit the memory_config, and the result is correct."""
    shard_shape = (1, 1, 64, 256)
    input_tensor, oracle_slices = _make_sharded_input(
        mesh_device, shard_shape, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    output_tensor = reduce_scatter(input_tensor, dim=3, topology=topology)
    ttnn.synchronize_device(mesh_device)
    assert output_tensor.memory_config().buffer_type == ttnn.BufferType.L1
    _check_outputs(mesh_device, output_tensor, oracle_slices, shard_shape, ttnn.bfloat16)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_max_slice_budget(mesh_device, topology):
    """Exactly AT the S = 256 resident-accumulator cliff (shard (1,1,256,4096) on
    N=4: P=1024, S=256): the largest cb_accumulator the Phase-0 op ever
    allocates (256 pages) must fit L1 alongside the streaming CBs and reduce
    correctly."""
    num_devices = _num_devices(mesh_device)
    if num_devices != 4:
        pytest.skip("shape is sized for the (1, 4) box (S = 256 exactly)")
    shard_shape = (1, 1, 256, 4096)
    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16)
    output_tensor = reduce_scatter(input_tensor, dim=3, topology=topology)
    ttnn.synchronize_device(mesh_device)
    _check_outputs(mesh_device, output_tensor, oracle_slices, shard_shape, ttnn.bfloat16)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_rejects_past_slice_budget(mesh_device, topology):
    """One tile-column past the cliff (S = 264 > 256) is rejected loudly with
    ValueError — the documented large-S refinement seam, never a silent OOM."""
    num_devices = _num_devices(mesh_device)
    if num_devices != 4:
        pytest.skip("shape is sized for the (1, 4) box (S = 264)")
    input_tensor, _ = _make_sharded_input(mesh_device, (1, 1, 256, 4224), ttnn.bfloat16)
    with pytest.raises(ValueError, match="accumulator budget"):
        reduce_scatter(input_tensor, dim=3, topology=topology)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_output_tensor_float32(mesh_device, topology):
    """float32 output_tensor path (acceptance covers bf16 only): writes into the
    supplied tensor and returns the same handle."""
    shard_shape = (1, 1, 64, 256)
    num_devices = _num_devices(mesh_device)
    out_shape = list(shard_shape)
    out_shape[3] //= num_devices
    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, ttnn.float32)
    preallocated = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.synchronize_device(mesh_device)
    returned = reduce_scatter(input_tensor, dim=3, topology=topology, output_tensor=preallocated)
    ttnn.synchronize_device(mesh_device)
    assert returned.buffer_address() == preallocated.buffer_address()
    _check_outputs(mesh_device, returned, oracle_slices, shard_shape, ttnn.float32)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_dim2_multibatch(mesh_device, topology):
    """dim=2 correctness with B > 1 (Refinement 2 Done-when): device i keeps
    rows [i*slice_H, (i+1)*slice_H) of EVERY batch plane — the reduce reader's
    per-batch walk restart on the golden (2,1,256,256) shape (a cursor hoisted
    out of the plane loop would silently read the wrong slice from batch 1 on)."""
    shard_shape = (2, 1, 256, 256)
    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16, dim=2)
    output_tensor = reduce_scatter(input_tensor, dim=2, topology=topology)
    ttnn.synchronize_device(mesh_device)
    _check_outputs(mesh_device, output_tensor, oracle_slices, shard_shape, ttnn.bfloat16, dim=2)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_loud_rejections(mesh_device, topology):
    """Host-side rejection edges: out-of-range dim and mismatched output_tensor
    spec raise ValueError (structural), not a typed axis refusal."""
    shard_shape = (1, 1, 64, 256)
    input_tensor, _ = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16)

    for bad_dim in (4, -5):
        with pytest.raises(ValueError, match="out of range"):
            reduce_scatter(input_tensor, dim=bad_dim, topology=topology)

    # Wrong-shape output_tensor (full shard shape instead of the 1/N slice).
    bad_output = ttnn.from_torch(
        torch.zeros(shard_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.synchronize_device(mesh_device)
    with pytest.raises(ValueError, match="output_tensor spec"):
        reduce_scatter(input_tensor, dim=3, topology=topology, output_tensor=bad_output)

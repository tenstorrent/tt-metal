# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Extended coverage for reduce_scatter_average (verifier-authored).

Focused gaps the acceptance/debug/golden suites do not touch:
  * dim=-1 negative alias (canonicalization BEFORE the SUPPORTED membership test)
  * L1-interleaved input/output (the design supports DRAM or L1 interleaved;
    every other suite runs DRAM)
  * float32 output_tensor path (acceptance covers only bf16)
  * typed refusals: out-of-SUPPORTED axis values raise UnsupportedAxisValue
    (a NotImplementedError subclass), NOT ValueError

Mesh-shape-adaptive via CCL_HW_MESH_SHAPE (default (1, 4)). Drive via:

    scripts/run_multidevice_sim_pytest.py --runtime hardware \
        --op reduce_scatter_average -- <this file>
"""

import os
from math import prod

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.reduce_scatter_average import reduce_scatter_average

MESH_SHAPE = tuple(int(x) for x in os.environ.get("CCL_HW_MESH_SHAPE", "1,4").split(","))

PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.99}
LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)


def _make_input(mesh_device, shard_shape, dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])
    torch.manual_seed(7)
    torch_full = torch.randn(full_shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        torch_full = torch_full.to(torch.bfloat16)
    shards = torch_full.reshape(num_devices, *shard_shape).to(torch.float32)
    mean = shards.mean(dim=0)
    oracle = list(mean.chunk(num_devices, dim=3))
    input_tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    return input_tensor, oracle


def _check(output_tensor, oracle, dtype, mesh_device):
    for dev_idx, t in enumerate(ttnn.get_device_tensors(output_tensor)):
        assert_with_pcc(oracle[dev_idx], ttnn.to_torch(t), PCC[dtype])


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_negative_dim_alias(mesh_device, topology):
    """dim=-1 canonicalizes to 3 before the SUPPORTED membership test."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("requires at least 2 mesh devices")
    input_tensor, oracle = _make_input(mesh_device, (1, 1, 64, 256), ttnn.bfloat16)
    output_tensor = reduce_scatter_average(input_tensor, dim=-1, topology=topology)
    ttnn.synchronize_device(mesh_device)
    _check(output_tensor, oracle, ttnn.bfloat16, mesh_device)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_l1_interleaved(mesh_device, topology, dtype):
    """L1-interleaved input (and derived L1 output + gather buffer)."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("requires at least 2 mesh devices")
    input_tensor, oracle = _make_input(mesh_device, (1, 1, 32, 256), dtype, ttnn.L1_MEMORY_CONFIG)
    output_tensor = reduce_scatter_average(input_tensor, topology=topology)
    ttnn.synchronize_device(mesh_device)
    assert output_tensor.memory_config().buffer_type == ttnn.BufferType.L1
    _check(output_tensor, oracle, dtype, mesh_device)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_output_tensor_fp32(mesh_device, topology):
    """The output_tensor path at float32 (acceptance covers only bf16)."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("requires at least 2 mesh devices")
    shard_shape = (1, 1, 64, 256)
    input_tensor, oracle = _make_input(mesh_device, shard_shape, ttnn.float32)
    out_shape = list(shard_shape)
    out_shape[3] //= num_devices
    preallocated = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.synchronize_device(mesh_device)
    returned = reduce_scatter_average(input_tensor, topology=topology, output_tensor=preallocated)
    ttnn.synchronize_device(mesh_device)
    assert returned.buffer_address() == preallocated.buffer_address()
    _check(returned, oracle, ttnn.float32, mesh_device)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_typed_refusals(mesh_device, topology):
    """Out-of-SUPPORTED axis values raise the registry-typed refusal
    (NotImplementedError subclass), not ValueError — validate() runs its axis
    gate BEFORE the axis-value-dependent structural checks."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("requires at least 2 mesh devices")
    input_tensor, _ = _make_input(mesh_device, (1, 1, 32, 256), ttnn.bfloat16)

    with pytest.raises(NotImplementedError):
        reduce_scatter_average(input_tensor, topology=ttnn.Topology.Ring)
    with pytest.raises(NotImplementedError):
        reduce_scatter_average(input_tensor, dim=2)
    # dim=-2 canonicalizes to 2, which is out of SUPPORTED — same typed refusal.
    with pytest.raises(NotImplementedError):
        reduce_scatter_average(input_tensor, dim=-2)

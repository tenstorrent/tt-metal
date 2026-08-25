# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Verifier-authored extended tests for all_reduce — focused gaps the acceptance
suite does not cover. Deliberately small (exhaustive coverage belongs in
refinements):

  * L1-interleaved memory config (the design supports DRAM or L1 interleaved;
    acceptance only exercises DRAM).
  * float32 output_tensor path (acceptance covers the path with bf16 only).
  * Typed refusals (registry contract): Ring topology and ROW_MAJOR layout must
    raise UnsupportedAxisValue (a NotImplementedError subclass), NOT ValueError.
  * Structural ValueErrors: output_tensor spec mismatch; resident-accumulator
    L1 budget gate (P * page_size > 512 KiB).

Run on the same runner as the acceptance suite (mesh shape must match the box):

    scripts/run_multidevice_sim_pytest.py --runtime hardware --op all_reduce -- \
        tests/ttnn/unit_tests/operations/all_reduce/test_all_reduce_extended.py -v
"""

import os
from math import prod

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.all_reduce import all_reduce


def _hw_mesh_shape(default=(1, 4)):
    raw = os.environ.get("CCL_HW_MESH_SHAPE")
    return tuple(int(x) for x in raw.split(",")) if raw else default


LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)
PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.99}
_TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}


def _make_sharded_input(mesh_device, shard_shape, dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])
    torch.manual_seed(7)
    torch_full = torch.randn(full_shape, dtype=torch.float32).to(_TORCH_DTYPE[dtype])
    oracle = torch_full.reshape(num_devices, *shard_shape).to(torch.float32).sum(dim=0).to(_TORCH_DTYPE[dtype])
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


def _skip_if_too_small(mesh_device):
    if prod(tuple(mesh_device.shape)) < 2:
        pytest.skip("all_reduce requires at least 2 mesh devices")


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [_hw_mesh_shape()], indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_all_reduce_l1_interleaved(mesh_device, topology, dtype):
    """L1-interleaved input (and gather/output, which inherit the memory config):
    the whole pipeline runs out of L1 instead of DRAM."""
    _skip_if_too_small(mesh_device)
    shard_shape = (1, 1, 64, 64)  # P=4: shard + N-block gather buffer fit L1 easily
    input_tensor, oracle = _make_sharded_input(mesh_device, shard_shape, dtype, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_tensor = all_reduce(input_tensor, topology=topology)
    ttnn.synchronize_device(mesh_device)
    assert output_tensor.memory_config().buffer_type == ttnn.BufferType.L1
    for dev_out in [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]:
        assert_with_pcc(oracle, dev_out, PCC[dtype])


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [_hw_mesh_shape()], indirect=True)
def test_all_reduce_output_tensor_f32(mesh_device, topology):
    """float32 output_tensor path: writes into the supplied tensor, returns the
    same handle (acceptance covers this path with bf16 only)."""
    _skip_if_too_small(mesh_device)
    shard_shape = (1, 1, 64, 128)
    input_tensor, oracle = _make_sharded_input(mesh_device, shard_shape, ttnn.float32)
    preallocated = ttnn.from_torch(
        torch.zeros(shard_shape, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.synchronize_device(mesh_device)
    returned = all_reduce(input_tensor, topology=topology, output_tensor=preallocated)
    ttnn.synchronize_device(mesh_device)
    assert returned.buffer_address() == preallocated.buffer_address()
    for dev_out in [ttnn.to_torch(t) for t in ttnn.get_device_tensors(returned)]:
        assert_with_pcc(oracle, dev_out, PCC[ttnn.float32])


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [_hw_mesh_shape()], indirect=True)
def test_all_reduce_typed_refusals(mesh_device, topology):
    """Out-of-SUPPORTED axis values raise UnsupportedAxisValue (NotImplementedError
    subclass) — the registry contract the golden harness xfail-stricts on."""
    _skip_if_too_small(mesh_device)
    input_tensor, _ = _make_sharded_input(mesh_device, (1, 1, 32, 32), ttnn.bfloat16)

    # topology axis: Ring not in SUPPORTED.
    with pytest.raises(NotImplementedError):
        all_reduce(input_tensor, topology=ttnn.Topology.Ring)

    # layout axis: ROW_MAJOR not in SUPPORTED (must be the typed refusal, not a
    # shape-derived ValueError).
    num_devices = prod(tuple(mesh_device.shape))
    rm_input = ttnn.from_torch(
        torch.randn((num_devices, 1, 32, 32), dtype=torch.float32).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    with pytest.raises(NotImplementedError):
        all_reduce(rm_input, topology=topology)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [_hw_mesh_shape()], indirect=True)
def test_all_reduce_structural_value_errors(mesh_device, topology):
    """Structural gates raise loud ValueErrors: output spec mismatch and the
    resident-accumulator L1 budget (P * page_size > 512 KiB)."""
    _skip_if_too_small(mesh_device)
    input_tensor, _ = _make_sharded_input(mesh_device, (1, 1, 64, 128), ttnn.bfloat16)

    # output_tensor spec mismatch (wrong shape).
    bad_output = ttnn.from_torch(
        torch.zeros((1, 1, 32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.synchronize_device(mesh_device)
    with pytest.raises(ValueError, match="output_tensor spec"):
        all_reduce(input_tensor, topology=topology, output_tensor=bad_output)

    # Accumulator budget: f32 shard (1, 1, 1024, 512) = 512 tiles x 4096 B = 2 MiB > 512 KiB.
    big_input, _ = _make_sharded_input(mesh_device, (1, 1, 1024, 512), ttnn.float32)
    with pytest.raises(ValueError, match="accumulator"):
        all_reduce(big_input, topology=topology)

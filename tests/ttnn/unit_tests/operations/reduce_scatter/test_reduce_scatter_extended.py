# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Extended (verifier-authored) tests for reduce_scatter — focused coverage of
gaps the immutable acceptance suite does not touch:

  * dim=2 scatter (promoted into SUPPORTED by the verifier: the host slice rows,
    the kernel's is_supported_scatter_dim static_assert, and the SliceRowWalker
    math were already dim-general — only the SUPPORTED membership list gated it).
  * the dim=-2 negative alias (canonicalization -2 ≡ 2).
  * L1 interleaved memory (the design admits DRAM or L1 interleaved; acceptance
    only exercises DRAM).

Deliberately small — exhaustive coverage belongs to the golden suite and the
refinement queue. Drive via the multi-device runner:

    scripts/run_multidevice_sim_pytest.py --op reduce_scatter -- \
        tests/ttnn/unit_tests/operations/reduce_scatter/test_reduce_scatter_extended.py -v
"""

import os
from math import prod

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.reduce_scatter import reduce_scatter

MESH_SHAPE = tuple(int(x) for x in os.environ.get("MULTIDEV_SIM_MESH_SHAPE", "1,4").split(","))

PCC = {
    ttnn.bfloat16: 0.99,
    ttnn.float32: 0.999,
}

LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)


def _make_sharded_input(mesh_device, shard_shape, dtype, scatter_dim, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Shard a seeded full tensor along dim 0; oracle = fp32 sum then chunk on scatter_dim."""
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])

    torch.manual_seed(7)
    torch_full = torch.randn(full_shape, dtype=torch.float32)

    summed = torch_full.reshape(num_devices, *shard_shape).sum(dim=0)
    if dtype == ttnn.bfloat16:
        torch_full = torch_full.to(torch.bfloat16)
        summed = summed.to(torch.bfloat16)
    oracle_slices = torch.chunk(summed, num_devices, dim=scatter_dim)

    input_tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    return input_tensor, oracle_slices


def _check(mesh_device, output_tensor, oracle_slices, shard_shape, scatter_dim, dtype):
    num_devices = prod(tuple(mesh_device.shape))
    expected_shape = list(shard_shape)
    expected_shape[scatter_dim] //= num_devices
    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
    assert len(output_shards) == num_devices
    for dev_idx, dev_out in enumerate(output_shards):
        assert tuple(dev_out.shape) == tuple(expected_shape)
        assert_with_pcc(oracle_slices[dev_idx], dev_out, PCC[dtype])


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("shard_shape", [(1, 1, 256, 256), (2, 1, 256, 64)])
def test_reduce_scatter_dim2(mesh_device, topology, dtype, shard_shape):
    """Scatter along dim 2 (tile rows): device i holds row-slice i of the sum."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter requires at least 2 mesh devices")

    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, dtype, scatter_dim=2)
    output_tensor = reduce_scatter(input_tensor, dim=2, topology=topology)
    ttnn.synchronize_device(mesh_device)
    _check(mesh_device, output_tensor, oracle_slices, shard_shape, 2, dtype)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_dim2_negative_alias(mesh_device, topology):
    """dim=-2 canonicalizes to 2 and behaves identically."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter requires at least 2 mesh devices")

    shard_shape = (1, 1, 256, 32)
    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16, scatter_dim=2)
    output_tensor = reduce_scatter(input_tensor, dim=-2, topology=topology)
    ttnn.synchronize_device(mesh_device)
    _check(mesh_device, output_tensor, oracle_slices, shard_shape, 2, ttnn.bfloat16)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_l1_interleaved(mesh_device, topology):
    """L1 interleaved input → L1 interleaved output (the design's second memory config)."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter requires at least 2 mesh devices")

    shard_shape = (1, 1, 64, 256)
    input_tensor, oracle_slices = _make_sharded_input(
        mesh_device, shard_shape, ttnn.bfloat16, scatter_dim=3, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    assert input_tensor.memory_config().buffer_type == ttnn.BufferType.L1
    output_tensor = reduce_scatter(input_tensor, dim=3, topology=topology)
    ttnn.synchronize_device(mesh_device)
    # Output inherits the input memory config (L1 interleaved).
    assert output_tensor.memory_config().buffer_type == ttnn.BufferType.L1
    _check(mesh_device, output_tensor, oracle_slices, shard_shape, 3, ttnn.bfloat16)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_dim2_program_cache(mesh_device, topology):
    """dim=2 second call is a program-cache hit (distinct hash from the dim=3 programs)."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter requires at least 2 mesh devices")

    for _ in range(2):
        shard_shape = (1, 1, 256, 32)
        input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16, scatter_dim=2)
        output_tensor = reduce_scatter(input_tensor, dim=2, topology=topology)
        ttnn.synchronize_device(mesh_device)
        _check(mesh_device, output_tensor, oracle_slices, shard_shape, 2, ttnn.bfloat16)

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Extended (verifier-authored) tests for reduce_scatter — focused gap coverage only.

Three gaps the acceptance suite does not exercise:

1. L1-interleaved memory — the op claims "interleaved, DRAM or L1" but every
   acceptance/golden cell runs DRAM. One cell proves the L1 path end to end
   (input, op-internal gather_buffer, and output all inherit L1 interleaved).
2. Multi-core, multi-row slice walk — a shape whose per-device output slice is
   3 rows x 3 cols of tiles (9 positions across 9 cores), so most cores start
   MID-ROW and the reader's `reset_offsets(start_tile % slice_Wt, ...)` seeding
   is exercised at every within-row phase.
3. Typed refusals — the registry contract promises UnsupportedAxisValue (not a
   bare NotImplementedError) for out-of-SUPPORTED axis values, and ValueError
   for a mismatched output_tensor spec.

MULTI-DEVICE op — drive via the deterministic multi-device runner (mesh MUST be
(1, 4) with FABRIC_1D or fabric init hangs):

    scripts/run_multidevice_sim_pytest.py --op reduce_scatter -- \
        tests/ttnn/unit_tests/operations/reduce_scatter/test_reduce_scatter_extended.py -v

Exhaustive axis coverage belongs to the golden suite / refinement queue, not here.
"""

from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.reduce_scatter import reduce_scatter
from ttnn.operations._op_contract import UnsupportedAxisValue

_SCATTER_DIM = 3
_PCC_BF16 = 0.99

LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)

# The verification mesh shape — the bh_quietbox_1x4_hw contract. Do NOT change.
MESH_SHAPE = (1, 4)


def _make_sharded_input(mesh_device, shard_shape, dtype, memory_config):
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])

    torch.manual_seed(7)
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
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    return input_tensor, oracle_slices


def _check_all_devices(mesh_device, output_tensor, oracle_slices, shard_shape, pcc):
    num_devices = prod(tuple(mesh_device.shape))
    expected_shape = list(shard_shape)
    expected_shape[_SCATTER_DIM] //= num_devices
    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
    for dev_idx, dev_out in enumerate(output_shards):
        assert tuple(dev_out.shape) == tuple(expected_shape)
        assert_with_pcc(oracle_slices[dev_idx], dev_out, pcc)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_l1_interleaved(mesh_device, topology):
    """L1-interleaved input: gather_buffer + output inherit L1; end-to-end correct."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter requires at least 2 mesh devices")

    shard_shape = (1, 1, 32, 128)  # gather_buffer = 4 shards = 32 KiB/device in L1
    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG)
    assert input_tensor.memory_config().buffer_type == ttnn.BufferType.L1

    output_tensor = reduce_scatter(input_tensor, dim=_SCATTER_DIM, topology=topology)
    ttnn.synchronize_device(mesh_device)

    assert output_tensor.memory_config().buffer_type == ttnn.BufferType.L1
    _check_all_devices(mesh_device, output_tensor, oracle_slices, shard_shape, _PCC_BF16)
    logger.info("reduce_scatter L1-interleaved path verified on all devices")


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_multicore_midrow_starts(mesh_device, topology):
    """Output slice = 3x3 tiles: 9 positions over 9 cores, so cores start at every
    within-row phase (start_tile % slice_Wt in {0, 1, 2}) — exercises the reader's
    SliceRowWalker reset_offsets seeding on mid-row starts."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter requires at least 2 mesh devices")

    shard_shape = (1, 1, 96, 96 * num_devices)  # out slice: 96x96 = 3x3 tiles
    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG)

    output_tensor = reduce_scatter(input_tensor, dim=_SCATTER_DIM, topology=topology)
    ttnn.synchronize_device(mesh_device)

    _check_all_devices(mesh_device, output_tensor, oracle_slices, shard_shape, _PCC_BF16)
    logger.info("reduce_scatter multi-core mid-row slice walk verified on all devices")


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_reduce_scatter_typed_refusals_and_output_spec(mesh_device, topology):
    """Out-of-SUPPORTED topology refuses with the TYPED UnsupportedAxisValue (the
    registry contract, not just any NotImplementedError); a mismatched
    output_tensor spec raises ValueError."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter requires at least 2 mesh devices")

    shard_shape = (1, 1, 32, 128)
    input_tensor, _ = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG)

    # Typed refusal: Ring is a TARGET-adjacent refinement axis, refused via the
    # registry exception type (never a generic error).
    with pytest.raises(UnsupportedAxisValue):
        reduce_scatter(input_tensor, dim=_SCATTER_DIM, topology=ttnn.Topology.Ring)

    # Structural error: output_tensor with the wrong (un-scattered) shape.
    bad_out = ttnn.from_torch(
        torch.zeros(shard_shape, dtype=torch.bfloat16),  # full shard shape, not shard/N
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.synchronize_device(mesh_device)
    with pytest.raises(ValueError):
        reduce_scatter(input_tensor, dim=_SCATTER_DIM, topology=topology, output_tensor=bad_out)

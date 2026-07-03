# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Extended, focused coverage for the all_gather CCL op.

The acceptance suite (test_all_gather.py) covers the happy path across four
shard shapes x {bf16, f32}. This module adds the two code paths it does NOT
exercise, kept deliberately small (the full axis matrix lives in the golden
suite, and axis expansion lives in the refinement queue):

  1. the caller-supplied ``output_tensor`` path (spec validation + reuse of a
     pre-allocated replicated output), and
  2. ``validate()`` rejection behaviour — the registry-model refusal
     (UnsupportedAxisValue for an out-of-SUPPORTED axis) and the structural
     ValueError guards (gather_dim out of range).

Run on the deterministic WH multi-device sim (mesh (1, 8) + FABRIC_1D):

    scripts/run_multidevice_sim_pytest.py --op all_gather -- \
        tests/ttnn/unit_tests/operations/all_gather/test_all_gather_extended.py -v
"""

from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations._op_contract import UnsupportedAxisValue
from ttnn.operations.all_gather import all_gather


LINEAR = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


def _make_sharded_input(mesh_device, shard_shape, dtype, layout=ttnn.TILE_LAYOUT):
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])
    torch.manual_seed(7)
    torch_full = torch.randn(full_shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        torch_full = torch_full.to(torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    return input_tensor, torch_full


@pytest.mark.parametrize("device_params", [LINEAR], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_all_gather_preallocated_output(mesh_device):
    """Passing a spec-matching, pre-allocated replicated output_tensor works and
    the op writes the full gather into it (covers the output-spec-validation +
    reuse branch that the acceptance suite never hits)."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("all_gather requires at least 2 mesh devices")

    shard_shape = (1, 1, 64, 128)
    input_tensor, torch_full = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16)

    # Build the resolved output spec by hand (gather_dim=0 scales dim 0 by N).
    out_shape = list(input_tensor.shape)
    out_shape[0] *= num_devices
    out_spec = ttnn.TensorSpec(
        ttnn.Shape(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.memory_config().buffer_type,
    )
    preallocated = ttnn.allocate_tensor_on_device(out_spec, mesh_device)

    returned = all_gather(input_tensor, 0, topology=ttnn.Topology.Linear, output_tensor=preallocated)
    ttnn.synchronize_device(mesh_device)

    # The op returns the very tensor we passed in (reuse, not a fresh alloc).
    assert returned is preallocated
    for dev_out in [ttnn.to_torch(t) for t in ttnn.get_device_tensors(returned)]:
        assert tuple(dev_out.shape) == tuple(torch_full.shape)
        assert_with_pcc(torch_full.to(torch.float32), dev_out.to(torch.float32), 0.995)
    logger.info("all_gather preallocated-output path: every device holds the full tensor")


@pytest.mark.parametrize("device_params", [LINEAR], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_all_gather_validate_rejections(mesh_device):
    """validate() gates: registry refusal for an out-of-SUPPORTED axis (Ring
    topology) and a structural ValueError for gather_dim out of range. Both fire
    before any fabric work, so this is fast (no gather).

    NOTE: ROW_MAJOR_LAYOUT and bfloat8_b were promoted to SUPPORTED in Refinement 1,
    so the still-unsupported axis exercised here is topology=Ring (a Refinement 3
    TARGET). gather_dim in {-3,-2,-1} is the other remaining refusal (Refinement 2)."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("all_gather requires at least 2 mesh devices")

    # Ring topology is not in SUPPORTED["topology"] -> typed registry refusal.
    tile_ring_input, _ = _make_sharded_input(mesh_device, (1, 1, 32, 64), ttnn.bfloat16)
    with pytest.raises(UnsupportedAxisValue):
        all_gather(tile_ring_input, 0, topology=ttnn.Topology.Ring)

    # gather_dim out of range (rank is 4) -> structural ValueError.
    tile_input, _ = _make_sharded_input(mesh_device, (1, 1, 32, 64), ttnn.bfloat16)
    with pytest.raises(ValueError):
        all_gather(tile_input, 9, topology=ttnn.Topology.Linear)
    logger.info("all_gather validate() rejections fire as expected (registry refusal + structural guard)")

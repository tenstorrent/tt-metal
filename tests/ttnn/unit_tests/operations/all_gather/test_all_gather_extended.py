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

from ttnn.operations._op_contract import ExcludedCell
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
    """validate() gates: registry refusal for a still-excluded CELL and a
    structural ValueError for gather_dim out of range. Both fire before any
    fabric work, so this is fast (no gather).

    NOTE: dtype {bf16,f32,bf8b}, layout {TILE,RM}, gather_dim {-4,-3,-2,-1}
    (Refinements 1-2) and topology {Linear,Ring} (Refinement 3) are ALL in
    SUPPORTED now, so no single-axis value is refused. The remaining refusal is
    the EXCLUSIONS cell {TILE, gather_dim=-2, non_tile_aligned} (a structural
    sub-tile-repack gap), which raises ExcludedCell (a NotImplementedError
    subclass)."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("all_gather requires at least 2 mesh devices")

    # topology=Ring is now SUPPORTED (Refinement 3) -> validate() must NOT raise.
    ring_input, _ = _make_sharded_input(mesh_device, (1, 1, 32, 64), ttnn.bfloat16)
    from ttnn.operations.all_gather.all_gather import validate as _validate

    _validate(ring_input, 0, topology=ttnn.Topology.Ring, output_tensor=None)  # no raise

    # EXCLUSIONS cell {TILE, gather_dim=-2, non_tile_aligned}: H=48 is not %32 in
    # (1,1,48,64), W=64 is tile-aligned -> gather_dim=-2 hits the excluded cell.
    excluded_input, _ = _make_sharded_input(mesh_device, (1, 1, 48, 64), ttnn.bfloat16)
    with pytest.raises(ExcludedCell):
        all_gather(excluded_input, -2, topology=ttnn.Topology.Linear)

    # gather_dim out of range (rank is 4) -> structural ValueError.
    tile_input, _ = _make_sharded_input(mesh_device, (1, 1, 32, 64), ttnn.bfloat16)
    with pytest.raises(ValueError):
        all_gather(tile_input, 9, topology=ttnn.Topology.Linear)
    logger.info("all_gather validate() rejections fire as expected (excluded cell + structural guard)")

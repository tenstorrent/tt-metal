# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Deterministic debugging tests for all_reduce. DO NOT DELETE — documents the
hand-calculable expectations used to verify the reduce pipeline.

Unlike the acceptance suite's torch.randn shards, every input here is
hand-calculable, so a wrong intermediate (mis-ordered contribution, wrong block
base in the gather_buffer, dropped arrival, misaligned add) shows up as an exact,
attributable value error rather than a PCC dip:

  * all-ones: every output element == N exactly (any positional shuffle of the
    adds is invisible, but a DROPPED or DOUBLE-counted contribution shifts the
    whole output by 1.0).
  * per-device constant (chip i holds i+1): output == N*(N+1)/2 everywhere.
    Distinguishes "some contribution missing" (which one: the deficit names the
    chip) from "all present".
  * index-encoded small integers: shard values are position-unique integers small
    enough to be EXACT in bf16 (< 256) and under fp32 FPU srcA/srcB truncation,
    so any tile reordering between contributions (an R11 positional-alignment
    break) produces a wrong value at a specific, decodable position.

Run (same runner as the acceptance suite — mesh shape must match the box):
    scripts/run_multidevice_sim_pytest.py --runtime hardware --op all_reduce -- \
        tests/ttnn/unit_tests/operations/all_reduce/test_all_reduce_debug.py
"""

import os
from math import prod

import pytest
import torch

import ttnn

from ttnn.operations.all_reduce import all_reduce

LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)


def _hw_mesh_shape(default=(1, 4)):
    raw = os.environ.get("CCL_HW_MESH_SHAPE")
    return tuple(int(x) for x in raw.split(",")) if raw else default


def _run_and_gather(mesh_device, torch_full, shard_shape, dtype, topology):
    """Shard torch_full along dim 0, run all_reduce, return per-device outputs."""
    input_tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    output_tensor = all_reduce(input_tensor, topology=topology)
    ttnn.synchronize_device(mesh_device)
    return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [_hw_mesh_shape()], indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_all_ones_single_tile(mesh_device, topology, dtype):
    """All-ones shards: every output element == N exactly (N=4: 4.0, exact in
    both dtypes). A dropped/doubled contribution shifts the whole output by 1."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("all_reduce requires at least 2 mesh devices")

    shard_shape = (1, 1, 32, 32)
    torch_full = torch.ones((num_devices, *shard_shape[1:]), dtype=torch.float32)
    expected = torch.full(shard_shape, float(num_devices), dtype=torch.float32)

    outputs = _run_and_gather(mesh_device, torch_full, shard_shape, dtype, topology)
    for dev_idx, dev_out in enumerate(outputs):
        assert torch.equal(dev_out.float(), expected), (
            f"device {dev_idx}: expected all {float(num_devices)}, "
            f"got min={dev_out.float().min()}, max={dev_out.float().max()}"
        )


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [_hw_mesh_shape()], indirect=True)
def test_per_device_constant_multitile(mesh_device, topology):
    """Chip i holds the constant i+1 (multi-tile shard, exercises g=4 granules):
    output == N*(N+1)/2 everywhere. A missing contribution's deficit names the
    chip that was dropped (e.g. sum 9 instead of 10 => chip 0 missing)."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("all_reduce requires at least 2 mesh devices")

    shard_shape = (1, 1, 64, 128)  # P=8, g=4
    blocks = [torch.full(shard_shape, float(i + 1), dtype=torch.float32) for i in range(num_devices)]
    torch_full = torch.cat(blocks, dim=0)
    total = num_devices * (num_devices + 1) / 2.0
    expected = torch.full(shard_shape, total, dtype=torch.float32)

    outputs = _run_and_gather(mesh_device, torch_full, shard_shape, ttnn.bfloat16, topology)
    for dev_idx, dev_out in enumerate(outputs):
        assert torch.equal(dev_out.float(), expected), (
            f"device {dev_idx}: expected all {total}, got unique values "
            f"{torch.unique(dev_out.float())[:8].tolist()}"
        )


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [_hw_mesh_shape()], indirect=True)
def test_index_encoded_positional_alignment(mesh_device, topology):
    """Position-unique small integers (exact in bf16: values < 256, sums < 256):
    shard element at flat index j on chip i is (j % 50) + i. Any tile reordering
    between contributions (an R11 break: a contribution walked in a different
    order than another) misaligns add_tiles operands and produces a decodably
    wrong value at the affected position. Expected: N*(j % 50) + N*(N-1)/2."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("all_reduce requires at least 2 mesh devices")

    shard_shape = (2, 1, 32, 64)  # multi-batch, P=4, g=4
    numel = prod(shard_shape)
    base = (torch.arange(numel, dtype=torch.float32) % 50).reshape(shard_shape)
    blocks = [base + float(i) for i in range(num_devices)]
    torch_full = torch.cat(blocks, dim=0)
    expected = num_devices * base + num_devices * (num_devices - 1) / 2.0

    outputs = _run_and_gather(mesh_device, torch_full, shard_shape, ttnn.bfloat16, topology)
    for dev_idx, dev_out in enumerate(outputs):
        mismatch = (dev_out.float() != expected).nonzero()
        assert mismatch.numel() == 0, (
            f"device {dev_idx}: {mismatch.shape[0]} mismatched positions; first at "
            f"{mismatch[0].tolist()} (expected {expected[tuple(mismatch[0])]}, "
            f"got {dev_out.float()[tuple(mismatch[0])]})"
        )

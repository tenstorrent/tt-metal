# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for #51133: from_torch(device=...) must not drop per-core allocation.

`create_tt_tensor_from_host_data` can build a tensor in its *source* layout on device and
convert it there with ttnn ops. Those ops rebuild the output MemoryConfig from a few named
fields and drop the experimental per-core allocation bit, so the caller silently received a
lockstep buffer.

The fix refuses the on-device construction path when the requested memory config asks for
per-core allocation, so the tensor is built on host and `to_device` applies the caller's
memory config directly — no op in between.
"""

import pytest
import torch

import ttnn
from conftest import requires_hybrid_allocator


# A single row of this many cores, used for the fixed-grid tests below. Spelling it once keeps
# the grid, the shard width and the tensor width from drifting apart.
NUM_CORES = 8


# Shapes from the sweep in #51133. The reporter established that core grid, core count and
# range count are all irrelevant — the only thing that mattered was whether the shard was
# small enough for has_sufficient_device_memory() to allow the on-device path. Grids here are
# kept narrow so the test runs on any device, unlike the col-12 grids in the report.
@requires_hybrid_allocator
@pytest.mark.parametrize(
    "grid_start, grid_end, shard_shape",
    [
        ((0, 0), (7, 0), (7168, 32)),  # gate_mm: the weight actually affected in tt-blaze
        ((0, 0), (7, 0), (512, 32)),  # small shard, same grid
        ((0, 0), (3, 0), (7168, 32)),  # 4 cores
        ((0, 0), (7, 1), (7168, 32)),  # 16 cores, two rows
    ],
    ids=["gate_mm_7168x32", "small_512x32", "four_cores", "two_rows"],
)
def test_from_torch_on_device_preserves_per_core_allocation(
    per_core_mesh_device, per_core_width_sharded_config, grid_start, grid_end, shard_shape
):
    """from_torch(device=...) must honour a per-core memory config, not downgrade to lockstep."""
    num_cores = (grid_end[0] - grid_start[0] + 1) * (grid_end[1] - grid_start[1] + 1)
    mem_config = per_core_width_sharded_config(grid_start, grid_end, shard_shape)

    torch_input = torch.randn(shard_shape[0], shard_shape[1] * num_cores, dtype=torch.bfloat16)
    tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mem_config,
        tile=ttnn.Tile((32, 32)),
        mesh_mapper=ttnn.ReplicateTensorToMesh(per_core_mesh_device),
        device=per_core_mesh_device,
    )

    assert tensor.is_per_core_allocated(), (
        "from_torch(device=...) silently dropped the experimental per-core allocation bit; "
        "the buffer is lockstep-allocated even though the memory_config asked for per-core"
    )
    assert torch.equal(ttnn.to_torch(tensor), torch_input), "from_torch corrupted the data"


@requires_hybrid_allocator
def test_from_torch_large_shard_still_per_core(per_core_mesh_device, per_core_width_sharded_config):
    """A shard above the has_sufficient_device_memory() gate already worked; keep it working.

    This case took the host path on `main` and so was unaffected by #51133. It is here to catch
    the fix breaking the branch that was already correct.
    """
    shard_height, shard_width = 14336, 32
    mem_config = per_core_width_sharded_config((0, 0), (NUM_CORES - 1, 0), (shard_height, shard_width))
    torch_input = torch.randn(shard_height, shard_width * NUM_CORES, dtype=torch.bfloat16)

    tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mem_config,
        tile=ttnn.Tile((32, 32)),
        mesh_mapper=ttnn.ReplicateTensorToMesh(per_core_mesh_device),
        device=per_core_mesh_device,
    )

    assert tensor.is_per_core_allocated()
    assert torch.equal(ttnn.to_torch(tensor), torch_input)


@requires_hybrid_allocator
def test_from_torch_row_major_per_core_unaffected(per_core_mesh_device, per_core_width_sharded_config):
    """ROW_MAJOR needs no layout conversion, so it always preserved the bit. Pin that."""
    shard_height, shard_width = 512, 32
    mem_config = per_core_width_sharded_config((0, 0), (NUM_CORES - 1, 0), (shard_height, shard_width))
    torch_input = torch.randn(shard_height, shard_width * NUM_CORES, dtype=torch.bfloat16)

    tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(per_core_mesh_device),
        device=per_core_mesh_device,
    )

    assert tensor.is_per_core_allocated()
    assert torch.equal(ttnn.to_torch(tensor), torch_input)

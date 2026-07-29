# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Ops must opt in before they can be handed a per-core allocated tensor.

Ops address a buffer by a single L1 address — `Buffer::address()` is the first core's, and
CB binding, runtime-arg patching and the host write/read all resolve through it (#51354). An
op that has not been taught to resolve per-core addresses would read every core as though it
shared the first core's allocation, which is silently wrong whenever those addresses differ.

`launch()` therefore refuses a per-core input to any op that has not declared
`supports_per_core_allocation`. Nothing declares it today, which matches the current state:
no op under ttnn/cpp/ttnn/operations resolves per-core addresses.
"""

import pytest
import torch

import ttnn
from conftest import requires_hybrid_allocator


def _width_sharded_config(grid_start, grid_end, shard_shape, per_core):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*grid_start), ttnn.CoreCoord(*grid_end))])
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs, list(shard_shape), ttnn.ShardOrientation.ROW_MAJOR),
    )
    if per_core:
        mem_config.experimental_set_per_core_allocation(True)
    return mem_config


@requires_hybrid_allocator
def test_op_rejects_per_core_input(per_core_mesh_device, expect_error):
    """Handing a per-core tensor to an op that has not opted in must fail loudly."""
    mem_config = _width_sharded_config((0, 0), (7, 0), (512, 32), per_core=True)
    torch_input = torch.randn(512, 32 * 8, dtype=torch.bfloat16)

    row_major = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(per_core_mesh_device),
        device=per_core_mesh_device,
    )
    assert row_major.is_per_core_allocated(), "precondition failed: ROW_MAJOR input is not per-core"

    with expect_error(RuntimeError, "has not opted in to per-core allocation"):
        ttnn.tilize(row_major, memory_config=mem_config)


@requires_hybrid_allocator
def test_lockstep_input_unaffected(per_core_mesh_device):
    """The check must be inert for ordinary lockstep tensors."""
    lockstep_config = _width_sharded_config((0, 0), (7, 0), (512, 32), per_core=False)
    torch_input = torch.randn(512, 32 * 8, dtype=torch.bfloat16)

    row_major = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=lockstep_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(per_core_mesh_device),
        device=per_core_mesh_device,
    )

    tiled = ttnn.tilize(row_major, memory_config=lockstep_config)
    assert torch.equal(ttnn.to_torch(tiled), torch_input), "tilize corrupted the data"

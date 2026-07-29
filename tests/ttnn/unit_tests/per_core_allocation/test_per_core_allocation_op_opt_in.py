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

import os

import pytest
import torch

import ttnn


@pytest.fixture(scope="function")
def per_core_mesh():
    """1x1 mesh with HYBRID allocator mode, which per-core allocation requires."""
    os.environ["TT_METAL_ALLOCATOR_MODE_HYBRID"] = "1"
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    yield mesh
    ttnn.close_mesh_device(mesh)
    os.environ.pop("TT_METAL_ALLOCATOR_MODE_HYBRID", None)


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


def test_op_rejects_per_core_input(per_core_mesh, expect_error):
    """Handing a per-core tensor to an op that has not opted in must fail loudly."""
    mem_config = _width_sharded_config((0, 0), (7, 0), (512, 32), per_core=True)
    torch_input = torch.randn(512, 32 * 8, dtype=torch.bfloat16)

    row_major = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(per_core_mesh),
        device=per_core_mesh,
    )
    assert row_major.is_per_core_allocated(), "precondition failed: ROW_MAJOR input is not per-core"

    with expect_error(RuntimeError, "has not opted in to per-core allocation"):
        ttnn.tilize(row_major, memory_config=mem_config)


def test_lockstep_input_unaffected(per_core_mesh):
    """The check must be inert for ordinary lockstep tensors."""
    lockstep_config = _width_sharded_config((0, 0), (7, 0), (512, 32), per_core=False)
    torch_input = torch.randn(512, 32 * 8, dtype=torch.bfloat16)

    row_major = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=lockstep_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(per_core_mesh),
        device=per_core_mesh,
    )

    tiled = ttnn.tilize(row_major, memory_config=lockstep_config)
    assert torch.equal(ttnn.to_torch(tiled), torch_input), "tilize corrupted the data"

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Layout ops must refuse a per-core allocated output rather than silently dropping the request.

`tilize`, `tilize_with_val_padding` and `untilize_with_unpadding` rebuild their output
MemoryConfig from a handful of named fields when the sharded program factory is selected.
Everything the caller asked for that is not named there is discarded, including the
experimental per-core allocation bit — the caller received a lockstep buffer with no signal
(the from_torch symptom of that is #51133).

These factories bind one L1 address for every core, so they could not honour a per-core
output even if the bit were carried through; see #51354. Refusing is therefore the correct
outcome, not a limitation to work around.
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


def test_tilize_rejects_per_core_output_config(per_core_mesh, expect_error):
    """A lockstep input with a per-core output request must fail loudly.

    The input is deliberately lockstep so this exercises the op's own guard on the requested
    output config, independent of any check on input tensors.
    """
    lockstep_config = _width_sharded_config((0, 0), (7, 0), (512, 32), per_core=False)
    per_core_config = _width_sharded_config((0, 0), (7, 0), (512, 32), per_core=True)
    torch_input = torch.randn(512, 32 * 8, dtype=torch.bfloat16)

    row_major = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=lockstep_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(per_core_mesh),
        device=per_core_mesh,
    )
    assert not row_major.is_per_core_allocated(), "precondition failed: input should be lockstep"

    with expect_error(RuntimeError, "per-core allocated output is not supported"):
        ttnn.tilize(row_major, memory_config=per_core_config)


def test_tilize_lockstep_output_unaffected(per_core_mesh):
    """The guard must be inert for ordinary lockstep use."""
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
    assert not tiled.is_per_core_allocated()
    assert torch.equal(ttnn.to_torch(tiled), torch_input), "tilize corrupted the data"

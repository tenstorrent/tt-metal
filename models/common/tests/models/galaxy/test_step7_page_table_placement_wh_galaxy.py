# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The one host assumption under Milestone B defect D-C1, settled on silicon.

`mb-coverage` attempt 1 derived D-C1 - *a prefill-shaped page table fed to
decode is accepted, not rejected* - entirely on the host, from a model of one
non-obvious ``ttnn`` fact: **a distributed tensor's ``.shape`` is the shard
shape, not the global one.** It read that out of
``ttnn/core/distributed/distributed_tensor.cpp`` and wrote, in its handoff:

    One line settles it. If it is ``(32, 64)`` instead, D-C1 is worse than
    described: the "device-local rows" branch would be unreachable for a
    correctly-mapped table, so decode's page table would have no effective
    validation at all.

This file is that line, as a test, because the difference decides how D-C1 is
written up. It needs a mesh and no model: three ``ttnn.from_torch`` calls, no
weights, no sub-devices, about twenty seconds.

What each case pins:

* the **decode** table is column-sharded (``ShardTensor2dMesh(dims=(None, 0))``
  over an ``(8, 4)`` mesh), so its device-local view is ``users_per_column`` = 8
  rows - one mesh column's users;
* the **prefill** table is replicated, so its device-local view is the whole
  physical batch, 32 rows. ``32 % 8 == 0``, which is precisely why
  ``Attention2D._validate_decode_page_table`` - which discriminates on row count
  alone - cannot tell it from a legitimate four-core L1 repeat;
* the two therefore differ in *placement*, not in shape, and placement is the
  discriminator the validator never consults.

Added by `mb-coverage` attempt 2, and run on a live 8x4 mesh.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.tests.models.galaxy.galaxy_hardware import (
    GALAXY_DEVICE_PARAMS,
    GALAXY_MESH_SHAPE,
    GALAXY_PHYSICAL_BATCH,
)

_USERS_PER_COLUMN = GALAXY_PHYSICAL_BATCH // GALAXY_MESH_SHAPE[1]
_BLOCKS = 64

_GALAXY = pytest.mark.parametrize("mesh_device", [pytest.param(GALAXY_MESH_SHAPE, id="8x4")], indirect=True)
_PARAMS = pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)


def _table() -> torch.Tensor:
    """A ``[32, 64]`` block-ownership table whose value encodes its own row."""

    return torch.arange(GALAXY_PHYSICAL_BATCH * _BLOCKS, dtype=torch.int32).reshape(GALAXY_PHYSICAL_BATCH, _BLOCKS)


def _stage(mesh_device: ttnn.MeshDevice, rows: torch.Tensor, *, sharded: bool):
    mapper = (
        ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=GALAXY_MESH_SHAPE)
        if sharded
        else ttnn.ReplicateTensorToMesh(mesh_device)
    )
    return ttnn.from_torch(
        rows,
        device=mesh_device,
        mesh_mapper=mapper,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@_PARAMS
@_GALAXY
def test_a_column_sharded_page_tables_device_local_view_is_one_columns_users(mesh_device: ttnn.MeshDevice):
    """``.shape`` is the shard shape: 8 rows, not 32."""

    table = _stage(mesh_device, _table(), sharded=True)
    try:
        shape = tuple(table.shape)
        print(f"[placement] decode table global=(32, {_BLOCKS}) device-local={shape}")
        assert shape == (_USERS_PER_COLUMN, _BLOCKS), (
            f"expected the shard shape ({_USERS_PER_COLUMN}, {_BLOCKS}); got {shape}. "
            "If this is (32, 64) then D-C1 is worse than reported: the decode validator's "
            "device-local-rows branch is unreachable for a correctly mapped table."
        )
    finally:
        ttnn.deallocate(table)


@_PARAMS
@_GALAXY
def test_a_replicated_page_tables_device_local_view_is_the_whole_physical_batch(mesh_device: ttnn.MeshDevice):
    """And 32 is a multiple of 8, which is the whole of D-C1."""

    table = _stage(mesh_device, _table(), sharded=False)
    try:
        shape = tuple(table.shape)
        print(f"[placement] prefill table global=(32, {_BLOCKS}) device-local={shape}")
        assert shape == (GALAXY_PHYSICAL_BATCH, _BLOCKS)
        assert shape[0] % _USERS_PER_COLUMN == 0, "the premise of D-C1 no longer holds"
    finally:
        ttnn.deallocate(table)


@_PARAMS
@_GALAXY
def test_the_two_placements_are_indistinguishable_by_row_count_but_not_by_memory_config(
    mesh_device: ttnn.MeshDevice,
):
    """The discriminator that exists, next to the one the validator uses.

    Both tables are DRAM-interleaved, so ``memory_config()`` alone does not
    separate them either - what separates them is *how many rows each device
    holds*, and that is exactly what ``.shape`` reports. A validator that
    compared the device-local row count against ``users_per_column`` **and
    rejected a strict multiple unless the tensor is L1 height-sharded** would
    separate them. The 32-row DRAM-interleaved case is the prefill layout; a
    32-row L1 height-sharded case is the legitimate four-core repeat.
    """

    sharded = _stage(mesh_device, _table(), sharded=True)
    replicated = _stage(mesh_device, _table(), sharded=False)
    try:
        rows_sharded, rows_replicated = tuple(sharded.shape)[0], tuple(replicated.shape)[0]
        assert rows_replicated % rows_sharded == 0
        print(
            f"[placement] decode rows={rows_sharded} prefill rows={rows_replicated} "
            f"ratio={rows_replicated // rows_sharded}  "
            f"decode memcfg={sharded.memory_config()}  prefill memcfg={replicated.memory_config()}"
        )
        assert sharded.memory_config().buffer_type == replicated.memory_config().buffer_type
        assert not sharded.memory_config().is_sharded()
        assert not replicated.memory_config().is_sharded()
    finally:
        ttnn.deallocate(sharded)
        ttnn.deallocate(replicated)

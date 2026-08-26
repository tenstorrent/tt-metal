# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""What the WH Galaxy `(8, 4)` decode partition actually looks like on silicon.

Milestone B's first hardware session lost several device runs to the same class
of defect: a decode-mode program placed on cores the loaded sub-device manager
does not own, or on dispatch cores. Both are properties of *this* mesh and this
dispatch configuration, and neither is knowable from a mocked mesh - the mock
supplies whatever ``compute_with_storage_grid_size`` the test author guessed.

So this file records the real numbers and asserts the invariants that the
placement recipes depend on. It is deliberately cheap: no checkpoint, no model,
no weights - just the mesh, its grid, and pure host arithmetic over
``recipes.py``. Running it takes about as long as opening the mesh.

`mb-qwen` and `mb-coverage` should run this first when a decode program aborts
with either of:

    TT_FATAL ... Kernel group cores do not match sub device cores
                 for programmable core type TENSIX
    TT_FATAL ... Illegal kernel placement for <kernel>,
                 Kernels cannot be placed on dispatch cores!
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.models.galaxy import recipes
from models.common.tests.models.galaxy.galaxy_hardware import GALAXY_DEVICE_PARAMS, GALAXY_MESH_SHAPE

galaxy = pytest.mark.parametrize("mesh_device", [pytest.param(GALAXY_MESH_SHAPE, id="8x4")], indirect=True)
galaxy_params = pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)


def _report(title: str, value: object) -> None:
    print(f"[partition] {title}: {value}", flush=True)


@galaxy_params
@galaxy
@torch.no_grad()
def test_worker_partition_matches_the_real_compute_grid(mesh_device):
    """Report the real grid, and pin the invariants the recipes rely on."""

    grid = mesh_device.compute_with_storage_grid_size()
    _report("compute_with_storage_grid_size", f"x={grid.x} y={grid.y}")
    _report("core_grid", mesh_device.core_grid)

    full = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    workers = recipes.worker_cores()
    senders = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in recipes.prefetch_sender_cores()])

    _report("worker_cores", f"{workers} ({workers.num_cores()} cores)")
    _report("prefetch senders", f"{senders} ({senders.num_cores()} cores)")
    _report("cores in the compute grid but in no partition", full.subtract(workers).subtract(senders))

    # The worker envelope has to be addressable, and must not overlap the senders.
    assert workers.subtract(full).num_cores() == 0, "worker_cores() leaves the compute grid"
    assert senders.subtract(full).num_cores() == 0, "prefetch senders leave the compute grid"
    overlap = workers.subtract(workers.subtract(senders))
    assert overlap.num_cores() == 0, f"worker_cores() overlaps the prefetch senders: {overlap}"


@galaxy_params
@galaxy
@torch.no_grad()
@pytest.mark.parametrize(
    "rows,local_k,local_n,name",
    [
        (32, 2048, 1280, "decode-qkv"),
        (32, 1024, 2048, "decode-wo"),
        (128, 2048, 1280, "prefill-128-qkv"),
        (2048, 1024, 2048, "prefill-2048-wo"),
    ],
)
def test_dense_matmul_work_grid_stays_inside_the_worker_partition(mesh_device, rows, local_k, local_n, name):
    """A dense mcast matmul must not reach a sender core or a dispatch core.

    ``MatmulMultiCoreReuseMultiCastProgramConfig`` anchors its work grid at
    ``allowed_worker_cores.bounding_box().start`` (or ``(0, 0)`` when the
    field is unset) and extends it by the number of output blocks, so the cores
    it will touch are computable on host from the program config alone. That is
    what this checks - the placement, not the arithmetic of the matmul.
    """

    config = recipes.dense_matmul_program_config(rows, local_k, local_n)
    grid = mesh_device.compute_with_storage_grid_size()
    workers = recipes.worker_cores()

    m_tiles = max(1, -(-rows // recipes.TILE))
    n_tiles = local_n // recipes.TILE
    blocks_y = -(-m_tiles // config.per_core_M)
    blocks_x = -(-n_tiles // config.per_core_N)
    start = (
        config.allowed_worker_cores.bounding_box().start
        if config.allowed_worker_cores is not None
        else ttnn.CoreCoord(0, 0)
    )
    used = ttnn.CoreRangeSet({ttnn.CoreRange(start, ttnn.CoreCoord(start.x + blocks_x - 1, start.y + blocks_y - 1))})
    _report(f"{name} allowed_worker_cores", config.allowed_worker_cores)
    _report(f"{name} work grid", f"{used} blocks_x={blocks_x} blocks_y={blocks_y}")

    assert start.x + blocks_x - 1 < grid.x, f"{name} work grid runs past the compute grid width {grid.x}"
    assert start.y + blocks_y - 1 < grid.y, f"{name} work grid runs past the compute grid height {grid.y}"
    outside = used.subtract(workers)
    assert outside.num_cores() == 0, f"{name} would run on {outside.num_cores()} non-worker core(s): {outside}"

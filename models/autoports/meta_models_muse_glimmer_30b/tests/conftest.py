# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared device fixtures for the DFlash tests.

The mesh lives here rather than in each test module because a session-scoped
fixture is per *module* only by convention, not by construction: two modules that
each define ``mesh_device`` both try to open a mesh in the same pytest process, and
the second open fails with

    Failed to open mesh device ... The device may be in an unrecoverable state and
    require a reset.

which reads like a hardware fault rather than the fixture collision it actually is.
Each file passed alone and 10 errored when run together, so this is worth
centralising: one mesh per session, shared by every module that asks for it.
"""

from __future__ import annotations

import gc

import pytest

import ttnn


@pytest.fixture(scope="session")
def mesh_device():
    """One 1x1 mesh for the whole test session.

    1x1 rather than the production 1x4 because these tests grade *maths*, and the
    drafter replicates its weights across the mesh, so a single die exercises the
    same kernels with a quarter of the load time.  Anything measuring throughput or
    sharding must open its own mesh at the real shape instead.
    """
    if ttnn.get_num_devices() < 1:  # pragma: no cover - no hardware
        pytest.skip("no Tenstorrent device available")
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)
        gc.collect()

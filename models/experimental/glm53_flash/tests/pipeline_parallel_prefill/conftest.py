# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Galaxy fixtures for the pipeline-parallel prefill tests: the 8 x 4 mesh opened once per session with a 1D fabric,
and its 1 x 4 row submeshes (one pipeline stage each)."""

from __future__ import annotations

import pytest

import ttnn

MESH_SHAPE = (8, 4)


@pytest.fixture(scope="session")
def galaxy():
    if ttnn.get_num_devices() < MESH_SHAPE[0] * MESH_SHAPE[1]:
        pytest.skip(f"needs a {MESH_SHAPE[0]} x {MESH_SHAPE[1]} Galaxy")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*MESH_SHAPE))
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="session")
def stages(galaxy):
    return [
        galaxy.create_submesh(ttnn.MeshShape(1, MESH_SHAPE[1]), ttnn.MeshCoordinate(r, 0)) for r in range(MESH_SHAPE[0])
    ]


@pytest.fixture(scope="session")
def stage(stages):
    return stages[0]

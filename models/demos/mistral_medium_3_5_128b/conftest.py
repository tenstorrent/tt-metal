# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Package fixtures: the 8x4 Galaxy mesh (SP=8 rows x TP=4 cols), opened once per session.

The mesh graph descriptor is exported here, at import, because it must be set before the cluster
initialises (see tt/fabric.py). Every device test uses the same session mesh so fabric bring-up is
paid once, not per test.
"""

import pytest

from models.demos.mistral_medium_3_5_128b.tt.fabric import apply_mesh_graph_descriptor

apply_mesh_graph_descriptor()

MESH_SHAPE = (8, 4)


@pytest.fixture(scope="session")
def galaxy_mesh():
    import ttnn
    from models.demos.mistral_medium_3_5_128b.tt.ccl import L1_SMALL_SIZE
    from models.demos.mistral_medium_3_5_128b.tt.fabric import fabric_config

    num_devices = ttnn.get_num_devices()
    if num_devices < MESH_SHAPE[0] * MESH_SHAPE[1]:
        pytest.skip(f"needs a 32-chip Galaxy, found {num_devices} devices")
    ttnn.set_fabric_config(fabric_config())
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*MESH_SHAPE), l1_small_size=L1_SMALL_SIZE)
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="session")
def ccl_manager(galaxy_mesh):
    from models.demos.mistral_medium_3_5_128b.tt.ccl import CCLManager
    from models.demos.mistral_medium_3_5_128b.tt.fabric import ccl_topology

    return CCLManager(galaxy_mesh, topology=ccl_topology())


@pytest.fixture(scope="session")
def mesh_config(galaxy_mesh):
    from models.demos.mistral_medium_3_5_128b.tt.ccl import MeshConfig

    return MeshConfig(galaxy_mesh.shape)

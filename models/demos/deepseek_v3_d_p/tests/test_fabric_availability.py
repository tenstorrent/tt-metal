# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Which fabrics this host can open at the production mesh shape.

A galaxy without wrap cabling opens FABRIC_2D but maps no torus, and the failure lands in topology
discovery rather than at runtime -- so a torus test fails during mesh open with a "could not fit in
the discovered physical topology" throw that reads like a broken test or a bad cable. Run this first
to tell the two apart: one line per fabric, and the torus rows fail on a host that is not
wrap-cabled.
"""

import pytest

import ttnn

MESH_SHAPE = (8, 4)

FABRICS = [
    ttnn.FabricConfig.FABRIC_2D,
    ttnn.FabricConfig.FABRIC_2D_TORUS_X,
    ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
    ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
]


@pytest.mark.parametrize("fabric_config", FABRICS, ids=lambda fabric: fabric.name)
def test_fabric_opens(fabric_config):
    num_devices = ttnn.get_num_devices()
    if num_devices != MESH_SHAPE[0] * MESH_SHAPE[1]:
        pytest.skip(f"{MESH_SHAPE} needs {MESH_SHAPE[0] * MESH_SHAPE[1]} devices; this host has {num_devices}")

    mesh_device = None
    try:
        ttnn.set_fabric_config(fabric_config)
        mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(*MESH_SHAPE))
        assert tuple(mesh_device.shape) == MESH_SHAPE
    except Exception as e:
        unmappable = "could not fit in the discovered physical topology" in str(e)
        reason = "host is not cabled for it (mesh-graph mapping failed)" if unmappable else str(e)
        pytest.fail(f"{fabric_config.name} cannot open {MESH_SHAPE}: {reason}")
    finally:
        if mesh_device is not None:
            ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Which fabrics this host can actually open at the production mesh shape.

A galaxy without wrap cabling opens FABRIC_2D but maps no torus, and the failure lands in topology
discovery rather than at runtime -- so a torus test fails during mesh open with a "could not fit in
the discovered physical topology" throw that reads like a broken test or a bad cable. Run this first
to tell the two apart: one line per fabric, and the torus rows fail on a host that is not
wrap-cabled.

A torus row can also fail WITHOUT throwing. The auto-discovery topology mapper falls back to a
lesser fabric type when the requested torus cannot be realized on the discovered cabling
(TORUS_XY -> TORUS_X, or TORUS_Y -> MESH), and open_mesh_device then SUCCEEDS at the requested
shape. Asserting the shape alone therefore reports PASS for a fabric the host cannot provide.
Verified on bh-glx-120-b07u08, which is X-wrap-only: pytest called it "4 passed" while the mapper
logged two downgrades. The mapper announces every fallback, so treat that warning as the verdict.
"""

import re

import pytest

import ttnn

MESH_SHAPE = (8, 4)

FABRICS = [
    ttnn.FabricConfig.FABRIC_2D,
    ttnn.FabricConfig.FABRIC_2D_TORUS_X,
    ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
    ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
]

# TopologyMapper::generate_mesh_graph_from_physical_system_descriptor logs this on every fallback.
DOWNGRADE = re.compile(r"requested fabric type (\S+) could not be realized.*?using (\S+)", re.S)


@pytest.mark.parametrize("fabric_config", FABRICS, ids=lambda fabric: fabric.name)
def test_fabric_opens(fabric_config, capfd):
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

    # The mesh opened at the right shape, but that does not mean the requested fabric was realized.
    # capfd is used rather than caplog because the mapper logs from C++, straight to the fd.
    captured = capfd.readouterr()
    downgrade = DOWNGRADE.search(captured.out + captured.err)
    if downgrade:
        pytest.fail(
            f"{fabric_config.name} was NOT realized on {MESH_SHAPE}: topology mapper fell back "
            f"{downgrade.group(1)} -> {downgrade.group(2)}. The mesh opened, so the shape assert "
            f"passed, but this host is not cabled for the requested torus."
        )

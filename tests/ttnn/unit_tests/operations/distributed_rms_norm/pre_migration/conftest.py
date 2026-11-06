# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The fabric mesh the pre-migration suite of distributed_rms_norm runs on.

distributed_rms_norm is a multi-device (CCL) op, so a one-chip `device` cannot run it. `drms_mesh` opens a mesh with a
fabric config and caches it across tests; it reopens only when the configuration changes. A test picks one with a
`mesh_cfg` parameter (default "system"):

    system   the whole system mesh, FABRIC_2D_TORUS_XY (the golden suite's deployment profile; Linear on any box)
    line     every device in one 1 x N line, FABRIC_2D_TORUS_XY (G = N; on a box without wrap links, the snake ring
             the golden ring mock uses, so Topology.Ring closes a real ring of N)
    line_1d  the same 1 x N line under FABRIC_1D (the 1D-only exchange schedule, and a fabric that does not wrap)
    ring_1d  the same 1 x N line under FABRIC_1D_RING (a 1D ring: every exchange schedule over Topology.Ring)

Items are sorted by configuration so each opens once per session.
"""

import os

import pytest
import ttnn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORDER = ("system", "line", "line_1d", "ring_1d")


def _system_shape():
    shape = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()
    return tuple(shape[i] for i in range(shape.dims()))


def mesh_config(name):
    """(mesh shape, fabric config) of a configuration name."""
    rows, cols = _system_shape()
    return {
        "system": ((rows, cols), ttnn.FabricConfig.FABRIC_2D_TORUS_XY),
        "line": ((1, rows * cols), ttnn.FabricConfig.FABRIC_2D_TORUS_XY),
        "line_1d": ((1, rows * cols), ttnn.FabricConfig.FABRIC_1D),
        "ring_1d": ((1, rows * cols), ttnn.FabricConfig.FABRIC_1D_RING),
    }[name]


def _mesh_cfg_of(item):
    callspec = getattr(item, "callspec", None)
    params = callspec.params if callspec is not None else {}
    name = params.get("mesh_cfg")
    if name is None:
        case = params.get("case")
        name = getattr(case, "mesh", "system")
    return name


def pytest_collection_modifyitems(items):
    slots = [i for i, item in enumerate(items) if str(item.path).startswith(_HERE)]
    mine = sorted((items[i] for i in slots), key=lambda item: _ORDER.index(_mesh_cfg_of(item)))
    for i, item in zip(slots, mine):
        items[i] = item


def _router_config():
    """The models' fabric router config: max packet payload 14 KiB (Blackhole) / 7 KiB (Wormhole) + 64 B."""
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    return cfg


class _MeshCache:
    def __init__(self):
        self.key = None
        self.mesh = None
        self.fabric_open = False

    def get(self, name):
        key = mesh_config(name)
        if self.mesh is not None and self.key == key:
            return self.mesh
        self.close()
        shape, fabric = key
        from tests.scripts.common import get_updated_device_params

        ttnn.set_fabric_config(
            fabric,
            ttnn.FabricReliabilityMode.STRICT_INIT,
            None,
            ttnn.FabricTensixConfig.DISABLED,
            ttnn.FabricUDMMode.DISABLED,
            ttnn.FabricManagerMode.DEFAULT,
            _router_config(),
        )
        self.fabric_open = True
        params = get_updated_device_params({"fabric_config": fabric})
        params.pop("fabric_config")
        self.mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*shape), **params)
        self.key = key
        return self.mesh

    def close(self):
        if self.mesh is not None:
            for submesh in self.mesh.get_submeshes():
                ttnn.close_mesh_device(submesh)
            ttnn.close_mesh_device(self.mesh)
            self.mesh = None
            self.key = None
        if self.fabric_open:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            self.fabric_open = False


@pytest.fixture(scope="session")
def _drms_mesh_cache():
    cache = _MeshCache()
    yield cache
    cache.close()


@pytest.fixture
def drms_mesh(request, _drms_mesh_cache):
    """The mesh of this test's `mesh_cfg` (or its `case.mesh`), "system" when it names none."""
    return _drms_mesh_cache.get(_mesh_cfg_of(request.node))

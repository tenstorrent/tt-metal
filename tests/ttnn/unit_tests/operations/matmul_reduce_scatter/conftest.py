# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Open the device once per module for this op's unit tests.

Applies @pytest.mark.use_module_device to every collected test; the root
`device` fixture is function-scoped and the marker switches it to module scope.
Do not define a local `device` fixture here — it shadows the root one and
disables the marker.
"""
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.use_module_device)


# --------------------------------------------------------------------------------------------
# Fabric meshes of the acceptance suite (test_acceptance.py, and the output_tensor / host-perf files)
# --------------------------------------------------------------------------------------------
# matmul_reduce_scatter is a multi-device Fabric op: it never runs on the one-chip `device` fixture. Each fixture
# below returns a mesh opened with the fabric config its placement needs, with the production router payload
# (14 KiB + 64 B on Blackhole, 7 KiB + 64 B on Wormhole), as the golden suite opens it. One mesh is open at a time:
# a session cache reopens only when the placement changes, so tests of one placement should sit together.
#   mrs_ring_mesh:  a ring of >= 3 devices under FABRIC_2D_TORUS_XY (the system mesh on a cluster with wrap links,
#                   else the box's snake ring opened as 1 x N); its group axis is the longer one.
#   mrs_line_mesh:  the system mesh under FABRIC_2D: Linear along every axis with >= 2 devices.
#   mrs_line4_mesh: the ring placement's shape under FABRIC_2D, which does not wrap it.


def _router_config():
    import ttnn

    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    return cfg


def _ring_shape():
    """The mesh shape of a real ring of >= 3 devices on this cluster, or None."""
    from eval.golden_tests.matmul_reduce_scatter.topology import ring_mock_meshes, system_mesh_shape, wrap_axes

    if wrap_axes():
        shape = system_mesh_shape()
        return shape if any(shape[a] >= 3 for a in wrap_axes()) else None
    mocks = ring_mock_meshes()
    return tuple(mocks[0][0]) if mocks else None


class _MrsMeshCache:
    def __init__(self):
        self.key = None
        self.mesh = None

    def get(self, mesh_shape, fabric_config):
        import ttnn
        from tests.scripts.common import get_updated_device_params

        key = (tuple(mesh_shape), fabric_config)
        if self.mesh is not None and self.key == key:
            return self.mesh
        self.close()
        ttnn.set_fabric_config(
            fabric_config,
            ttnn.FabricReliabilityMode.STRICT_INIT,
            None,
            ttnn.FabricTensixConfig.DISABLED,
            ttnn.FabricUDMMode.DISABLED,
            ttnn.FabricManagerMode.DEFAULT,
            _router_config(),
        )
        params = get_updated_device_params({"fabric_config": fabric_config})
        params.pop("fabric_config")
        self.mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape), **params)
        self.key = key
        return self.mesh

    def close(self):
        import ttnn

        if self.mesh is not None:
            for submesh in self.mesh.get_submeshes():
                ttnn.close_mesh_device(submesh)
            ttnn.close_mesh_device(self.mesh)
            self.mesh = None
        if self.key is not None:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            self.key = None


@pytest.fixture(scope="session")
def _mrs_mesh_cache():
    cache = _MrsMeshCache()
    yield cache
    cache.close()


@pytest.fixture
def mrs_ring_mesh(_mrs_mesh_cache):
    import ttnn

    shape = _ring_shape()
    if shape is None:
        pytest.skip("this cluster cannot close a ring of >= 3 devices")
    return _mrs_mesh_cache.get(shape, ttnn.FabricConfig.FABRIC_2D_TORUS_XY)


@pytest.fixture
def mrs_line_mesh(_mrs_mesh_cache):
    import ttnn
    from eval.golden_tests.matmul_reduce_scatter.topology import system_mesh_shape

    shape = system_mesh_shape()
    if max(shape) < 2:
        pytest.skip("matmul_reduce_scatter needs >= 2 devices")
    return _mrs_mesh_cache.get(shape, ttnn.FabricConfig.FABRIC_2D)


@pytest.fixture
def mrs_line4_mesh(_mrs_mesh_cache):
    import ttnn

    shape = _ring_shape()
    if shape is None:
        pytest.skip("this cluster has no group of >= 3 devices")
    return _mrs_mesh_cache.get(shape, ttnn.FabricConfig.FABRIC_2D)

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Open the device once per module for this op's unit tests.

Applies @pytest.mark.use_module_device to every collected test; the root
`device` fixture is function-scoped and the marker switches it to module scope.
Do not define a local `device` fixture here — it shadows the root one and
disables the marker.

matmul_reduce_scatter is a multi-device Fabric op, so the pre-migration suite (test_acceptance.py,
test_matmul_reduce_scatter_output_tensor.py, test_matmul_reduce_scatter_host_perf.py) opens meshes instead:
  mrs_line_mesh            the system mesh under FABRIC_2D (Linear along every axis with >= 2 devices)
  mrs_ring_mesh            a real ring: the system mesh under FABRIC_2D_TORUS_XY on a cluster with wrap links
                           (Galaxy), else the box's snake ring opened as 1 x N under FABRIC_2D_TORUS_XY (QuietBox 2 /
                           LoudBox)
  mrs_ring_axis            the cluster_axis that ring closes along
  mrs_unwrapped_ring_mesh  the ring placement's shape under plain FABRIC_2D (no wrap: a Ring of >= 3 is refused)
  mrs_mesh                 "line" | "ring" by indirect parametrization
Every mesh uses the production router payload (14 KiB + 64 B on Blackhole), as the golden suite does. One mesh is
open at a time (two fabric placements cannot be open together); the items using them are grouped by placement within
each module, so each placement opens once per module.
"""
import sys

import pytest
import ttnn


_PLACEMENT_RANK = {"mrs_line_mesh": 0, "mrs_unwrapped_ring_mesh": 1, "mrs_ring_mesh": 2}


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.use_module_device)
    # Group each module's mesh items by placement (stable), keeping the modules where they were.
    slots = [i for i, item in enumerate(items) if _PLACEMENT_RANK.keys() & set(item.fixturenames)]
    first = {}
    for i in slots:
        first.setdefault(str(items[i].path), i)
    ordered = sorted(
        (items[i] for i in slots),
        key=lambda item: (
            first[str(item.path)],
            max(_PLACEMENT_RANK.get(name, 0) for name in item.fixturenames),
        ),
    )
    for i, item in zip(slots, ordered):
        items[i] = item


TRACE_REGION_BYTES = 64 << 20  # the host-perf spec's traced state captures several calls


def _router_config():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    return cfg


def _ring_placement():
    """(mesh_shape, fabric_config, cluster_axis) of a real ring on this cluster, or None."""
    from eval.golden_tests.matmul_reduce_scatter.topology import ring_mock_meshes, system_mesh_shape, wrap_axes

    if wrap_axes():
        shape = system_mesh_shape()
        axis = next((a for a in wrap_axes() if shape[a] >= 3), None)
        return None if axis is None else (shape, ttnn.FabricConfig.FABRIC_2D_TORUS_XY, axis)
    mocks = ring_mock_meshes()
    if not mocks:
        return None
    shape, axis, _ = mocks[0]
    return shape, ttnn.FabricConfig.FABRIC_2D_TORUS_XY, axis


def _precompile_pass():
    """The wrapper's precompile plugin runs every test body once under NO_DISPATCH before the real pass. The op's
    per-mesh state (global semaphores, a zeroed mailbox) made then never reached the device, so the real pass must not
    reuse a mesh the collect pass used: the pass is part of the mesh key."""
    return getattr(sys.modules.get("tests.plugins.up_front_collect"), "_PASS", None)


class _MeshCache:
    def __init__(self):
        self.key = None
        self.mesh = None

    def get(self, mesh_shape, fabric_config):
        key = (tuple(mesh_shape), fabric_config, _precompile_pass())
        if self.mesh is not None and self.key == key:
            return self.mesh
        self.close()
        from tests.scripts.common import get_updated_device_params

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
        params["trace_region_size"] = TRACE_REGION_BYTES
        self.mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape), **params)
        self.key = key
        return self.mesh

    def close(self):
        if self.mesh is not None:
            for submesh in self.mesh.get_submeshes():
                ttnn.close_mesh_device(submesh)
            ttnn.close_mesh_device(self.mesh)
            self.mesh = None
            self.key = None
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def _mrs_mesh_cache():
    cache = _MeshCache()
    yield cache
    cache.close()


@pytest.fixture
def mrs_line_mesh(_mrs_mesh_cache):
    from eval.golden_tests.matmul_reduce_scatter.topology import system_mesh_shape

    return _mrs_mesh_cache.get(system_mesh_shape(), ttnn.FabricConfig.FABRIC_2D)


@pytest.fixture
def mrs_ring_axis():
    placement = _ring_placement()
    if placement is None:
        pytest.skip("this cluster cannot close a ring")
    return placement[2]


@pytest.fixture
def mrs_ring_mesh(_mrs_mesh_cache, mrs_ring_axis):
    shape, fabric_config, _ = _ring_placement()
    return _mrs_mesh_cache.get(shape, fabric_config)


@pytest.fixture
def mrs_unwrapped_ring_mesh(_mrs_mesh_cache, mrs_ring_axis):
    shape, _, _ = _ring_placement()
    return _mrs_mesh_cache.get(shape, ttnn.FabricConfig.FABRIC_2D)


@pytest.fixture
def mrs_mesh(request):
    """The placement an indirect parametrization names ("line" | "ring"), opened at setup (never in a test body,
    which the precompile pass runs under NO_DISPATCH)."""
    if request.param == "line":
        return request.getfixturevalue("mrs_line_mesh")
    if request.param == "ring":
        return request.getfixturevalue("mrs_ring_mesh")
    raise pytest.UsageError(f"mrs_mesh: unknown placement {request.param!r}")

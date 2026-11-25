# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures of the all_gather_rms_norm pre-migration suite.

``mesh_device`` (shadowing the root fixture for this directory only): a mesh opened under FABRIC_2D_TORUS_XY with the
production fabric packet payload (14 KiB + 64 B on Blackhole, 7 KiB + 64 B on Wormhole), program cache enabled,
cached across tests and reopened only when the requested shape changes. Request a shape with
``@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)``; the default is (1, 4): on a 4-chip box
without wrap links that is the box's snake ring, the 4-device group the models run (cluster_axis 1). (4, 1) is the
same group along cluster_axis 0.

Items of this directory are sorted by mesh shape so every shape opens once per session.
"""

from __future__ import annotations

import gc
import os
import sys

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.all_gather_rms_norm.pre_migration.agrn_cases import RegistryGroup

FABRIC_CONFIG = ttnn.FabricConfig.FABRIC_2D_TORUS_XY
DEFAULT_MESH_SHAPE = (1, 4)
_HERE = os.path.dirname(os.path.abspath(__file__))


def _router_config():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    return cfg


def _requested_shape(item):
    callspec = getattr(item, "callspec", None)
    if callspec is not None and "mesh_device" in callspec.params:
        param = callspec.params["mesh_device"]
        if isinstance(param, RegistryGroup):  # sort key only: the 1xN / Nx1 group of its axis
            return (1, 4) if param[1] == 1 else (4, 1)
        return tuple(param)
    return DEFAULT_MESH_SHAPE


def pytest_collection_modifyitems(items):
    slots = [i for i, item in enumerate(items) if str(item.path).startswith(_HERE)]
    mine = sorted((items[i] for i in slots), key=_requested_shape)
    for i, item in zip(slots, mine):
        items[i] = item


class _FabricMeshCache:
    def __init__(self):
        self.shape = None
        self.mesh = None
        self.fabric_open = False

    def get(self, shape):
        shape = tuple(shape)
        if self.mesh is not None and self.shape == shape:
            return self.mesh
        self.close()
        from tests.scripts.common import get_updated_device_params

        rows, cols = shape
        available = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape().mesh_size()
        if rows * cols > available:
            pytest.skip(f"mesh {shape} needs {rows * cols} devices, this system has {available}")
        ttnn.set_fabric_config(
            FABRIC_CONFIG,
            ttnn.FabricReliabilityMode.STRICT_INIT,
            None,
            ttnn.FabricTensixConfig.DISABLED,
            ttnn.FabricUDMMode.DISABLED,
            ttnn.FabricManagerMode.DEFAULT,
            _router_config(),
        )
        self.fabric_open = True
        params = get_updated_device_params({"fabric_config": FABRIC_CONFIG})
        params.pop("fabric_config")
        self.mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols), **params)
        self.mesh.enable_program_cache()
        self.shape = shape
        return self.mesh

    def close(self):
        if self.mesh is not None:
            for submesh in self.mesh.get_submeshes():
                ttnn.close_mesh_device(submesh)
            ttnn.close_mesh_device(self.mesh)
            self.mesh = None
            self.shape = None
        if self.fabric_open:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            self.fabric_open = False


@pytest.fixture(scope="session")
def _agrn_mesh_cache():
    cache = _FabricMeshCache()
    yield cache
    cache.close()


@pytest.fixture
def mesh_device(request, _agrn_mesh_cache):
    shape = getattr(request, "param", None) or DEFAULT_MESH_SHAPE
    if isinstance(shape, RegistryGroup):
        from eval.golden_tests.all_gather_rms_norm.topology import registry_mesh

        resolved, why = registry_mesh(*shape)
        if resolved is None:
            pytest.skip(f"not runnable on this cluster: {why}")
        shape = resolved
    return _agrn_mesh_cache.get(shape)


@pytest.fixture(autouse=True)
def _release_failed_test_tensors():
    """pytest keeps a failed test's traceback (sys.last_*), and with it that test's device tensors: L1 buffers left
    alive that way change the next test's L1 picture (allocation failures, CB-fit refusals). Drop them after every
    test so each case starts from the same allocator state."""
    yield
    for name in ("last_type", "last_value", "last_traceback", "last_exc"):
        if hasattr(sys, name):
            setattr(sys, name, None)
    gc.collect()

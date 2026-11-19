# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Opt-in reuse of one open mesh device across the Gemma 4 unit tests of a process.

Enabled with ``GEMMA4_TEST_MESH_REUSE=1`` (the 31B unit leg of ``tests/pipeline_reorg/models_unit_tests.yaml``).
Opening a mesh costs seconds per device test, which is most of a short test's time. With reuse, the ``mesh_device``
fixture in this directory's conftest hands consecutive tests the same open mesh while the mesh shape and
``device_params`` stay the same, and collection groups the device tests by that key. A test gets a newly opened mesh
when the key changes, after any failed test, when the previous test left submeshes open, when its ``device_params``
reserve a trace region (a trace it did not release would fill the next test's region), or when it is marked
``gemma4_fresh_mesh``. A test that captures into the mesh's implicit trace region releases its traces in a
``finally``: the next test on the mesh gets that region. The model's state keyed by mesh id
(``tt/experts/moe_decode._SHARED``) is dropped when the mesh passes to the next test. The mesh closes before a test
outside this directory and at session end. Same pattern as ``models/demos/deepseek_v3_d_p/tests/_reuse.py``.
"""

import os
import sys

from loguru import logger

REUSE_ENV = "GEMMA4_TEST_MESH_REUSE"
FRESH_MARKER = "gemma4_fresh_mesh"
_UNIT_DIR = "gemma4/tests/unit/"


def reuse_enabled() -> bool:
    return os.environ.get(REUSE_ENV) == "1"


def _freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return str(value)


def mesh_key(param, device_params):
    """What a reused mesh must match: the requested mesh (``request.param``) and the parametrized ``device_params``."""
    return (_freeze(param), _freeze(device_params or {}))


def _forget_mesh_state(device):
    """Drop the model's module-level state keyed by this mesh's id, so the next test builds its own."""
    moe_decode = sys.modules.get("models.demos.gemma4.tt.experts.moe_decode")
    if moe_decode is not None:
        moe_decode._SHARED.pop(int(device.id()), None)


class _Registry:
    def __init__(self):
        self.device = None
        self.gen = None
        self.key = None
        self.dirty = False

    def get_device(self, request, silicon_arch_name, device_params):
        from conftest import mesh_device as root_mesh_device

        # Read before the root fixture opens the mesh: it writes the resolved trace_region_size into device_params
        # (the root device_params fixture's per-test copy).
        key = mesh_key(getattr(request, "param", None), device_params)
        fresh = request.node.get_closest_marker(FRESH_MARKER) is not None or bool(
            (device_params or {}).get("trace_region_size")
        )
        if self.dirty or fresh or (self.device is not None and (key != self.key or self.device.get_submeshes())):
            self.close()
        if self.device is not None:
            _forget_mesh_state(self.device)
            logger.info("reusing open mesh device")
            return self.device
        gen = root_mesh_device.__wrapped__(request, silicon_arch_name, device_params)
        self.device = next(gen)
        self.gen = gen
        self.key = key
        if fresh:
            self.dirty = True
        return self.device

    def close(self):
        self.dirty = False
        if self.device is None:
            return
        gen, self.device, self.gen, self.key = self.gen, None, None, None
        next(gen, None)


_REGISTRY = _Registry()


def get_device(request, silicon_arch_name, device_params):
    return _REGISTRY.get_device(request, silicon_arch_name, device_params)


def mark_dirty():
    """Close the mesh after this test (fixture finalizers may still use it now)."""
    _REGISTRY.dirty = True


def after_test(nextitem):
    """Close the mesh after a failed or fresh-mesh test, and before a test outside this directory or the session end."""
    if _REGISTRY.dirty or nextitem is None or _UNIT_DIR not in nextitem.nodeid:
        _REGISTRY.close()


def close():
    _REGISTRY.close()


def group_by_mesh(items):
    """Order this directory's tests so tests sharing a mesh key run back to back; host tests first, in their order.

    Only the positions this directory's tests already hold are permuted; other tests keep their places.
    """
    positions = [i for i, item in enumerate(items) if _UNIT_DIR in item.nodeid]

    def sort_key(item):
        if "mesh_device" not in getattr(item, "fixturenames", ()):
            return (0, "")
        params = getattr(getattr(item, "callspec", None), "params", {})
        return (1, repr(mesh_key(params.get("mesh_device"), params.get("device_params"))))

    ordered = sorted((items[i] for i in positions), key=sort_key)
    for i, item in zip(positions, ordered):
        items[i] = item

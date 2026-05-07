# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Reusable shared-fixture helpers for TTNN pytest conftests.

This module factors out the patterns that emerged across
``tests/ttnn/unit_tests/operations/eltwise/conftest.py`` (module-scoped
``_DeviceManager``) and ``tests/nightly/t3000/ccl/conftest.py``
(session-scoped ``_MeshDeviceManager``) so that per-directory conftests
can share the same well-tested logic.

Three manager classes are provided:

- :class:`DeviceManager` — module-scoped single-device manager. Equivalent
  to the pattern used by eltwise / sdpa / deepseek / tensor.
- :class:`ParamKeyedDeviceManager` — session-scoped device manager that
  auto-detects compatible runs across files in the same directory. When a
  test requests a device with the same ``device_params`` as the cached
  one, the existing device is reused; when params change, the device is
  closed and reopened. This is the recommended pattern for directories
  whose tests parametrize ``device_params`` (e.g. conv, matmul, pool,
  data_movement).
- :class:`ParamKeyedMeshDeviceManager` — session-scoped mesh manager
  keyed on ``(mesh_shape, device_params)``. Mirrors the T3K-CCL design.

Per-directory conftests typically follow this template::

    import pytest
    from tests.ttnn.conftest_helpers import (
        ParamKeyedDeviceManager,
        register_device_markers,
        resolve_device_id,
        sort_items_by_device_params,
    )

    def pytest_configure(config):
        register_device_markers(config)

    @pytest.fixture(scope="session")
    def _device_manager(request):
        mgr = ParamKeyedDeviceManager(resolve_device_id(request.config))
        yield mgr
        mgr.close()

    @pytest.fixture(scope="function")
    def device(request, _device_manager, device_params):
        yield _device_manager.get(device_params, request)

    def pytest_collection_modifyitems(config, items):
        sort_items_by_device_params(items)

The collection-ordering hook batches adjacent tests with the same
``device_params`` so the manager-cached device is reused.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Repo-root resolution and root-conftest loader
# ---------------------------------------------------------------------------
#
# The repo-root conftest exposes helpers (`is_tg_cluster`, `first_available_tg_device`,
# `set_fabric`, `reset_fabric`) that we re-use here.  pytest does not let us
# import the root conftest by name (it's loaded as `conftest`), so we mimic
# the pattern used by other directory conftests: re-load the root conftest
# under a private module name so its helpers are importable without the
# import-time side effects of pytest's own conftest plugin.


def _repo_root() -> Path:
    for env in ("TT_METAL_HOME", "TT_METAL_ROOT"):
        v = os.environ.get(env)
        if v:
            return Path(v).resolve()
    # tests/ttnn/conftest_helpers.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


_ROOT = _repo_root()
_ROOT_CONFTEST_MODULE = "tt_metal_root_conftest_for_conftest_helpers"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _root_conftest():
    """Load repo-root conftest under a private module name."""
    if _ROOT_CONFTEST_MODULE in sys.modules:
        return sys.modules[_ROOT_CONFTEST_MODULE]
    path = _ROOT / "conftest.py"
    spec = importlib.util.spec_from_file_location(_ROOT_CONFTEST_MODULE, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_ROOT_CONFTEST_MODULE] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------


def register_device_markers(config) -> None:
    """Register the device-lifetime markers used by the manager fixtures."""
    config.addinivalue_line(
        "markers",
        "requires_fresh_device: test needs a dedicated per-test device (empty program cache)",
    )
    config.addinivalue_line(
        "markers",
        "manages_own_device: test opens its own device internally (e.g. ttnn.manage_device) "
        "and must not have a fixture device open concurrently",
    )


def register_mesh_device_markers(config) -> None:
    """Register the mesh-device-lifetime markers used by the mesh-manager fixtures."""
    config.addinivalue_line(
        "markers",
        "requires_fresh_mesh_device: test needs a dedicated per-test mesh device",
    )


# ---------------------------------------------------------------------------
# Device-id resolution
# ---------------------------------------------------------------------------


def resolve_device_id(config) -> int:
    """Return the device_id from CLI options, accounting for TG clusters."""
    rc = _root_conftest()
    device_id = config.getoption("device_id")
    if rc.is_tg_cluster() and not device_id:
        device_id = rc.first_available_tg_device()
    return device_id


# ---------------------------------------------------------------------------
# Single-device helpers
# ---------------------------------------------------------------------------


def _params_key(device_params: dict | None) -> tuple:
    """Return a hashable key uniquely identifying a device_params dict."""
    if not device_params:
        return ()
    return tuple(sorted((k, str(v)) for k, v in device_params.items()))


def _open_device(device_id: int, device_params: dict | None = None):
    """Open a single device with optional params and attach a CacheEntriesCounter."""
    import ttnn
    from tests.scripts.common import get_updated_device_params
    from tests.tests_common.cache_entries_counter import CacheEntriesCounter

    updated = get_updated_device_params(dict(device_params or {}))
    # Strip mesh/fabric-only keys that ttnn.CreateDevice does not accept.
    for key in (
        "fabric_config",
        "fabric_tensix_config",
        "reliability_mode",
        "fabric_manager",
        "fabric_router_config",
        "dispatch_core_axis",
    ):
        updated.pop(key, None)
    dev = ttnn.CreateDevice(device_id=device_id, **updated)
    ttnn.SetDefaultDevice(dev)
    dev.cache_entries_counter = CacheEntriesCounter(dev)
    return dev


def _close_device(dev) -> None:
    import ttnn

    ttnn.close_device(dev)


# ---------------------------------------------------------------------------
# Module-scoped DeviceManager (eltwise / sdpa / deepseek pattern)
# ---------------------------------------------------------------------------


class DeviceManager:
    """Module-scoped shared device with optional fresh-device suspend/resume.

    Use this when every test in the module shares a single device configuration
    (no ``device_params`` parametrization).  For directories that parametrize
    ``device_params``, use :class:`ParamKeyedDeviceManager` instead.
    """

    def __init__(self, device_id: int, label: str = "shared"):
        self.device = None
        self.device_id = device_id
        self._label = label

    def get(self):
        if self.device is None:
            self.device = _open_device(self.device_id)
            logger.info("DeviceManager[{}] opened device_id={}", self._label, self.device_id)
        return self.device

    def suspend(self) -> None:
        """Temporarily close the shared device so a fresh device can use the hardware."""
        if self.device is not None:
            _close_device(self.device)
            logger.info("DeviceManager[{}] suspended for fresh-device test", self._label)
            self.device = None

    def resume(self) -> None:
        """No-op: the device is lazily reopened on next ``get()`` call."""
        pass

    def close(self) -> None:
        if self.device is not None:
            _close_device(self.device)
            logger.info("DeviceManager[{}] closed", self._label)
            self.device = None


# ---------------------------------------------------------------------------
# Param-keyed DeviceManager (auto-detects compatible runs)
# ---------------------------------------------------------------------------


class ParamKeyedDeviceManager:
    """Session-scoped device manager keyed by ``device_params``.

    The first ``get(device_params)`` call opens a device with those params.
    Subsequent calls with the **same** params return the cached device
    without any open/close.  When ``device_params`` change, the cached
    device is closed and a new one is opened.

    This auto-detects compatible runs across files in the same directory:
    parametrized tests that happen to share ``device_params`` keys batch
    automatically without any per-file conftest changes.

    Combine with :func:`sort_items_by_device_params` (called from
    ``pytest_collection_modifyitems``) to group adjacent tests by their
    ``device_params`` key for maximum reuse.
    """

    def __init__(self, device_id: int, label: str = "param_keyed"):
        self.device = None
        self.device_id = device_id
        self._params_key: tuple = ()
        self._label = label

    def get(self, device_params: dict | None, request=None):
        """Return (or open) a device matching the requested ``device_params``.

        ``request`` is accepted for parity with the mesh manager and is
        currently unused for the single-device path.
        """
        key = _params_key(device_params)
        if self.device is not None and self._params_key == key:
            return self.device

        # Different params (or first call) -- close and reopen.
        if self.device is not None:
            logger.info("ParamKeyedDeviceManager[{}] params changed; reopening device", self._label)
            _close_device(self.device)
            self.device = None

        self.device = _open_device(self.device_id, device_params)
        self._params_key = key
        logger.info(
            "ParamKeyedDeviceManager[{}] opened device_id={} params_key={}",
            self._label,
            self.device_id,
            key,
        )
        return self.device

    def suspend(self) -> None:
        """Close the cached device so a fresh-device test can use the hardware."""
        if self.device is not None:
            _close_device(self.device)
            logger.info("ParamKeyedDeviceManager[{}] suspended for fresh-device test", self._label)
            self.device = None
            self._params_key = ()

    def resume(self) -> None:
        """No-op: the device is lazily reopened on next ``get()`` call."""
        pass

    def close(self) -> None:
        if self.device is not None:
            _close_device(self.device)
            logger.info("ParamKeyedDeviceManager[{}] closed", self._label)
            self.device = None
            self._params_key = ()


# ---------------------------------------------------------------------------
# Mesh-device helpers (T3K-CCL pattern)
# ---------------------------------------------------------------------------


def _resolve_mesh_param(request) -> tuple:
    """Resolve the requested mesh shape from request.param into (MeshShape, tuple).

    Returns ``(mesh_shape_obj, mesh_shape_tuple)``.  When request.param is
    absent, defaults to the system mesh size.
    """
    import ttnn
    import pytest

    try:
        param = request.param
    except (ValueError, AttributeError):
        param = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape().mesh_size()

    if isinstance(param, tuple):
        grid_dims = param
        if len(grid_dims) != 2:
            raise ValueError("Device mesh grid shape must have exactly two elements.")
        num_devices_requested = grid_dims[0] * grid_dims[1]
        if not ttnn.using_distributed_env() and num_devices_requested > ttnn.get_num_devices():
            pytest.skip("Requested more devices than available. Test not applicable for machine")
        return ttnn.MeshShape(*grid_dims), grid_dims

    if not ttnn.using_distributed_env() and param > ttnn.get_num_devices():
        pytest.skip("Requested more devices than available. Test not applicable for machine")
    return ttnn.MeshShape(1, param), (1, param)


def _open_mesh_device(mesh_shape_obj, device_params: dict, request):
    """Open a mesh device with fabric configuration applied."""
    import ttnn
    from tests.scripts.common import get_updated_device_params
    from tests.tests_common.cache_entries_counter import CacheEntriesCounter

    rc = _root_conftest()
    actual_params = dict(device_params)
    updated = get_updated_device_params(actual_params)

    fabric_config = updated.pop("fabric_config", None)
    fabric_tensix_config = updated.pop("fabric_tensix_config", None)
    reliability_mode = updated.pop("reliability_mode", None)
    fabric_manager = updated.pop("fabric_manager", None)
    fabric_router_config = updated.pop("fabric_router_config", None)

    rc.set_fabric(fabric_config, reliability_mode, fabric_tensix_config, fabric_manager, fabric_router_config)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape_obj, **updated)
    mesh_device.cache_entries_counter = CacheEntriesCounter(mesh_device)
    return mesh_device, fabric_config


def _close_mesh_device(mesh_device, fabric_config) -> None:
    import ttnn

    rc = _root_conftest()
    if mesh_device is None:
        return
    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh_device)
    rc.reset_fabric(fabric_config)


# ---------------------------------------------------------------------------
# Param-keyed MeshDeviceManager (T3K-CCL style)
# ---------------------------------------------------------------------------


class ParamKeyedMeshDeviceManager:
    """Session-scoped mesh device manager keyed by (mesh_shape, device_params).

    Mirrors the design of ``tests/nightly/t3000/ccl/conftest.py`` and is
    the recommended pattern for any directory whose tests open multi-device
    meshes with multiple distinct configurations.
    """

    def __init__(self, label: str = "param_keyed_mesh"):
        self.mesh_device = None
        self._config_key: tuple = ()
        self._fabric_config = None
        self._label = label

    def get(self, device_params: dict, request):
        mesh_shape_obj, mesh_shape_tuple = _resolve_mesh_param(request)
        key = (mesh_shape_tuple, _params_key(device_params))

        if self.mesh_device is not None and self._config_key == key:
            return self.mesh_device

        # Different config (or first call) -- close and reopen.
        self._close()

        self.mesh_device, self._fabric_config = _open_mesh_device(mesh_shape_obj, device_params, request)
        self._config_key = key
        logger.info(
            "ParamKeyedMeshDeviceManager[{}] opened shape={} fabric={}",
            self._label,
            mesh_shape_tuple,
            self._fabric_config,
        )
        return self.mesh_device

    def _close(self) -> None:
        if self.mesh_device is not None:
            _close_mesh_device(self.mesh_device, self._fabric_config)
            logger.info("ParamKeyedMeshDeviceManager[{}] closed cached mesh", self._label)
            self.mesh_device = None
            self._config_key = ()
            self._fabric_config = None

    def close(self) -> None:
        self._close()


# ---------------------------------------------------------------------------
# Per-test cleanup helpers
# ---------------------------------------------------------------------------


def reset_device_state(device) -> None:
    """Clean per-test residue on a shared device handle.

    Clears loaded sub-device manager (when not in slow dispatch mode) and
    resets the cache_entries_counter.  Does NOT clear the program cache --
    that breaks ethernet CQ init on T3K.
    """
    if device is None:
        return
    if not os.environ.get("TT_METAL_SLOW_DISPATCH_MODE"):
        try:
            device.clear_loaded_sub_device_manager()
        except Exception:
            logger.exception("clear_loaded_sub_device_manager failed during reset")
    try:
        device.cache_entries_counter.reset()
    except Exception:
        logger.exception("cache_entries_counter.reset failed during reset")


def reset_mesh_device_state(mesh_device) -> None:
    """Clean per-test residue on a shared mesh handle (T3K-CCL pattern)."""
    if mesh_device is None:
        return
    try:
        mesh_device.clear_loaded_sub_device_manager()
    except Exception:
        logger.exception("clear_loaded_sub_device_manager failed during mesh reset")
    try:
        mesh_device.reset_sub_device_stall_group()
    except Exception:
        logger.exception("reset_sub_device_stall_group failed during mesh reset")
    try:
        mesh_device.cache_entries_counter.reset()
    except Exception:
        logger.exception("cache_entries_counter.reset failed during mesh reset")
    try:
        mesh_device.enable_program_cache()
    except Exception:
        logger.exception("enable_program_cache failed during mesh reset")


# ---------------------------------------------------------------------------
# Test ordering hook
# ---------------------------------------------------------------------------


def _item_device_params_key(item) -> tuple:
    """Best-effort extraction of the device_params parametrization from a pytest item.

    Used by :func:`sort_items_by_device_params` to group tests by their
    requested device params so the param-keyed manager can keep its cached
    device across adjacent tests.

    Returns the same key shape as :func:`_params_key`: a tuple of sorted
    (key, str(value)) pairs, or ``()`` if the test does not parametrize
    ``device_params``.
    """
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return ()
    params = getattr(callspec, "params", {})
    raw = params.get("device_params")
    if raw is None:
        return ()
    if isinstance(raw, dict):
        return _params_key(raw)
    # Some tests pass non-dict objects (e.g. enum values); use their string form.
    return ((str(raw),),)


def sort_items_by_device_params(items) -> None:
    """Stable-sort ``items`` so adjacent tests share their device_params key.

    Call from a directory conftest's ``pytest_collection_modifyitems`` hook::

        def pytest_collection_modifyitems(config, items):
            sort_items_by_device_params(items)

    This keeps the relative order of tests with the same ``device_params``
    while batching adjacent groups so the param-keyed manager keeps its
    cached device across runs of the same config.
    """
    items.sort(key=_item_device_params_key)


# ---------------------------------------------------------------------------
# Re-exports of root conftest helpers (for downstream conftests that need them)
# ---------------------------------------------------------------------------


def root_conftest() -> Any:
    """Return the loaded root conftest module (lazy)."""
    return _root_conftest()

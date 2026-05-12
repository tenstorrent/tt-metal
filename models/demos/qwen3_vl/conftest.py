# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Session-scoped ``mesh_device`` fixture for the qwen3_vl T3K test suite.

All eight files under ``models/demos/qwen3_vl/tests/`` request the same
``(1, 8)`` T3K mesh with ``device_params={"fabric_config": True}``.
The repo-root ``mesh_device`` fixture is function-scoped and pays
``open_mesh_device`` + ``SetFabricConfig(...)`` + ``close_mesh_device``
+ ``SetFabricConfig(DISABLED)`` per test.  Across the CI gate that
expands to one open/close per parametrize case in every test file —
several minutes of pure setup overhead the suite does not need.

This conftest overrides the root ``mesh_device`` fixture with one that
delegates to a session-scoped ``_MeshDeviceManager``.  The manager
keeps a single mesh open across all tests that share the same
``(mesh_shape, device_params)`` tuple and only re-opens when a test
asks for a different config.  Because every qwen3_vl test asks for the
same tuple, the cache hit rate is 100% inside this directory.

Design mirrors ``tests/nightly/t3000/ccl/conftest.py`` (proven CCL
pattern) and ``tests/ttnn/unit_tests/operations/eltwise/conftest.py``
(``requires_fresh_device`` / ``manages_own_device`` opt-out markers).

Tests that intentionally need a fresh device can still opt out with::

    @pytest.mark.requires_fresh_device

Tests that open / close devices themselves should use::

    @pytest.mark.manages_own_device
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
from loguru import logger

from models.tt_transformers.conftest import device_params  # noqa: F401  (re-exported for tests)

# ---------------------------------------------------------------------------
# Root-conftest loader
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    for env in ("TT_METAL_HOME", "TT_METAL_ROOT"):
        v = os.environ.get(env)
        if v:
            return Path(v).resolve()
    return Path(__file__).resolve().parents[3]


_ROOT = _repo_root()
_ROOT_CONFTEST_MODULE = "tt_metal_root_conftest_for_qwen3_vl"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _root_conftest():
    """Load repo-root conftest helpers without clashing with pytest's own import."""
    if _ROOT_CONFTEST_MODULE in sys.modules:
        return sys.modules[_ROOT_CONFTEST_MODULE]
    path = _ROOT / "conftest.py"
    spec = importlib.util.spec_from_file_location(_ROOT_CONFTEST_MODULE, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_ROOT_CONFTEST_MODULE] = mod
    spec.loader.exec_module(mod)
    return mod


_rc = _root_conftest()


# ---------------------------------------------------------------------------
# Custom markers
# ---------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_fresh_device: test needs a dedicated per-test mesh device "
        "(closes the cached qwen3_vl mesh and re-opens for the next test).",
    )
    config.addinivalue_line(
        "markers",
        "manages_own_device: test opens its own device internally and must not "
        "have a fixture mesh device open concurrently.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config_key(mesh_shape_tuple, device_params_dict):
    """Hashable key uniquely identifying a mesh device config."""
    dp_key = tuple(sorted((k, str(v)) for k, v in device_params_dict.items()))
    return (mesh_shape_tuple, dp_key)


# ---------------------------------------------------------------------------
# Session-scoped mesh device manager
# ---------------------------------------------------------------------------


class _MeshDeviceManager:
    """Caches a single mesh device across tests with matching config."""

    def __init__(self):
        self.mesh_device = None
        self._config_key = None
        self._fabric_config = None

    def get(self, device_params_dict, request):
        import ttnn
        from models.tt_transformers.demo.trace_region_config import get_supported_trace_region_size
        from tests.scripts.common import get_updated_device_params

        try:
            param = request.param
        except (ValueError, AttributeError):
            param = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape().mesh_size()

        if isinstance(param, tuple):
            grid_dims = param
            assert len(grid_dims) == 2, "Device mesh grid shape should have exactly two elements."
            num_devices_requested = grid_dims[0] * grid_dims[1]
            if not ttnn.using_distributed_env() and num_devices_requested > ttnn.get_num_devices():
                pytest.skip("Requested more devices than available. Test not applicable for machine")
            mesh_shape_obj = ttnn.MeshShape(*grid_dims)
            mesh_shape_tuple = grid_dims
        else:
            if not ttnn.using_distributed_env() and param > ttnn.get_num_devices():
                pytest.skip("Requested more devices than available. Test not applicable for machine")
            mesh_shape_obj = ttnn.MeshShape(1, param)
            mesh_shape_tuple = (1, param)

        override_trace_region_size = get_supported_trace_region_size(request, param)
        actual_device_params = dict(device_params_dict)
        if override_trace_region_size:
            actual_device_params["trace_region_size"] = override_trace_region_size
            logger.info(f"Overriding trace region size to {override_trace_region_size}")

        updated_device_params = get_updated_device_params(actual_device_params)
        fabric_config = updated_device_params.pop("fabric_config", None)
        fabric_tensix_config = updated_device_params.pop("fabric_tensix_config", None)
        reliability_mode = updated_device_params.pop("reliability_mode", None)
        fabric_manager = updated_device_params.pop("fabric_manager", None)
        fabric_router_config = updated_device_params.pop("fabric_router_config", None)

        key = _config_key(mesh_shape_tuple, actual_device_params)

        if self.mesh_device is not None and self._config_key == key:
            logger.debug("qwen3_vl: reusing cached mesh device (config unchanged)")
            return self.mesh_device

        self._close()

        _rc.set_fabric(fabric_config, reliability_mode, fabric_tensix_config, fabric_manager, fabric_router_config)
        mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape_obj, **updated_device_params)

        from tests.tests_common.cache_entries_counter import CacheEntriesCounter

        mesh_device.cache_entries_counter = CacheEntriesCounter(mesh_device)

        self.mesh_device = mesh_device
        self._config_key = key
        self._fabric_config = fabric_config
        logger.info(
            "qwen3_vl: opened new mesh device shape={} fabric={}",
            mesh_shape_tuple,
            fabric_config,
        )
        return mesh_device

    def _close(self):
        import ttnn

        if self.mesh_device is not None:
            for submesh in self.mesh_device.get_submeshes():
                ttnn.close_mesh_device(submesh)
            ttnn.close_mesh_device(self.mesh_device)
            _rc.reset_fabric(self._fabric_config)
            logger.info("qwen3_vl: closed cached mesh device")
            self.mesh_device = None
            self._config_key = None
            self._fabric_config = None

    def close(self):
        self._close()


# ---------------------------------------------------------------------------
# Session-scoped manager fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _mesh_device_manager():
    mgr = _MeshDeviceManager()
    yield mgr
    mgr.close()


# ---------------------------------------------------------------------------
# Function-scoped mesh_device fixture (overrides root conftest)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def mesh_device(request, silicon_arch_name, device_params, _mesh_device_manager):
    """
    qwen3_vl mesh_device override.

    Routes through the session-scoped manager so consecutive tests with
    the same ``(mesh_shape, device_params)`` reuse one mesh open.

    Opt-out markers (mirroring eltwise conftest):
      ``@pytest.mark.requires_fresh_device`` — close the manager's mesh
      before this test runs; the test then uses the root function-scoped
      open/close path via a fresh manager cache miss.
      ``@pytest.mark.manages_own_device`` — test opens its own device;
      this fixture is skipped entirely.
    """
    import ttnn

    if request.node.get_closest_marker("manages_own_device"):
        yield None
        return

    if request.node.get_closest_marker("requires_fresh_device"):
        _mesh_device_manager._close()

    request.node.pci_ids = ttnn.get_pcie_device_ids()

    mesh_dev = _mesh_device_manager.get(device_params, request)

    yield mesh_dev


# ---------------------------------------------------------------------------
# Per-test mesh device state reset (autouse)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_mesh_device_state(request, _mesh_device_manager):
    """
    Best-effort cleanup of mutable per-mesh state after each test so the
    cached device behaves cleanly for the next test.  Does NOT close the
    device.  Does NOT call clear_program_cache() (breaks ETH CQ init).
    """
    yield

    md = _mesh_device_manager.mesh_device
    if md is None:
        return

    try:
        md.clear_loaded_sub_device_manager()
    except Exception:
        logger.exception("qwen3_vl: failed to clear loaded sub-device manager")
    try:
        md.reset_sub_device_stall_group()
    except Exception:
        logger.exception("qwen3_vl: failed to reset sub-device stall group")
    try:
        md.cache_entries_counter.reset()
    except Exception:
        logger.exception("qwen3_vl: failed to reset cache_entries_counter")
    try:
        md.enable_program_cache()
    except Exception:
        logger.exception("qwen3_vl: failed to re-enable program cache")

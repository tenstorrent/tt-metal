# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Session-scoped ``mesh_device`` fixture for T3K CCL tests.

Overrides the root function-scoped ``mesh_device`` fixture so that a
single 8-device mesh is opened once and reused across all CCL tests that
share the same ``(mesh_shape, device_params)`` configuration.  When a
test requests a different configuration the old mesh is torn down and a
new one is opened — but consecutive tests with the same config skip the
expensive open/close cycle entirely.

Each individual pytest invocation of the CCL test suite opens and closes
the mesh device cluster many times.  With ~14 separate ``pytest`` calls
in ``run_t3000_ccl_tests``, batching them into fewer calls (and reusing
the mesh within each call) eliminates the majority of redundant
``open_mesh_device`` / ``close_mesh_device`` round-trips.

Design mirrors the ``_DeviceManager`` pattern from
``tests/ttnn/unit_tests/operations/eltwise/conftest.py``, adapted for
multi-device meshes with fabric configuration.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
from loguru import logger


# ---------------------------------------------------------------------------
# Root-conftest loader
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    for env in ("TT_METAL_HOME", "TT_METAL_ROOT"):
        v = os.environ.get(env)
        if v:
            return Path(v).resolve()
    return Path(__file__).resolve().parents[4]


_ROOT = _repo_root()
_ROOT_CONFTEST_MODULE = "tt_metal_root_conftest_for_ccl"

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
# Helpers
# ---------------------------------------------------------------------------

def _config_key(mesh_shape_tuple, device_params):
    """Return a hashable key that uniquely identifies a mesh device config."""
    # Sort device_params for determinism; values must be converted to
    # something hashable (fabric enums, ints, etc. are fine as str).
    dp_key = tuple(sorted((k, str(v)) for k, v in device_params.items()))
    return (mesh_shape_tuple, dp_key)


# ---------------------------------------------------------------------------
# Session-scoped mesh device manager
# ---------------------------------------------------------------------------

class _MeshDeviceManager:
    """
    Caches a single mesh device across tests.  When the requested config
    matches the cached one the device is returned as-is.  When the config
    changes the old mesh is torn down and a new one opened.

    This avoids re-opening an 8-device mesh for every single test function
    while still supporting tests that need different fabric configs.
    """

    def __init__(self):
        self.mesh_device = None
        self._config_key = None
        self._fabric_config = None

    def get(self, mesh_shape, device_params, request):
        """Return (or open) a mesh device matching the requested config."""
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
        actual_device_params = dict(device_params)
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
            logger.info("Reusing cached mesh device (config unchanged)")
            return self.mesh_device

        # Config changed — tear down old mesh and open a new one
        self._close()

        _rc.set_fabric(fabric_config, reliability_mode, fabric_tensix_config, fabric_manager, fabric_router_config)
        mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape_obj, **updated_device_params)

        from tests.tests_common.cache_entries_counter import CacheEntriesCounter
        mesh_device.cache_entries_counter = CacheEntriesCounter(mesh_device)

        self.mesh_device = mesh_device
        self._config_key = key
        self._fabric_config = fabric_config
        logger.info(
            "Opened new mesh device: shape={}, fabric={}",
            mesh_shape_tuple,
            fabric_config,
        )
        return mesh_device

    def _close(self):
        """Close the currently cached mesh device (if any)."""
        import ttnn

        if self.mesh_device is not None:
            for submesh in self.mesh_device.get_submeshes():
                ttnn.close_mesh_device(submesh)
            ttnn.close_mesh_device(self.mesh_device)
            _rc.reset_fabric(self._fabric_config)
            logger.info("Closed cached mesh device")
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
    """Session-scoped mesh device manager — lives for the entire pytest session."""
    mgr = _MeshDeviceManager()
    yield mgr
    mgr.close()


# ---------------------------------------------------------------------------
# Function-scoped mesh_device fixture (overrides root conftest)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def mesh_device(request, silicon_arch_name, device_params, _mesh_device_manager):
    """
    Per-test mesh_device fixture for CCL tests.

    Delegates to the session-scoped ``_MeshDeviceManager`` which keeps
    the mesh open across tests that share the same configuration.  The
    mesh is only torn down and reopened when a test requests a different
    ``(mesh_shape, device_params)`` combination.

    ``silicon_arch_name`` is accepted (but unused) so that pytest's
    ``pytest_generate_tests`` hook adds the architecture prefix to test
    IDs (e.g. ``[wormhole_b0-...]``), keeping IDs compatible with the
    shell-script invocations in ``run_t3000_ccl_tests``.

    Compatible with ``@pytest.mark.parametrize("mesh_device", [...], indirect=True)``
    and ``@pytest.mark.parametrize("device_params", [...], indirect=True)``.
    """
    import ttnn

    # pci_ids must be set at function scope for pytest teardown logging
    request.node.pci_ids = ttnn.get_pcie_device_ids()

    mesh_dev = _mesh_device_manager.get(None, device_params, request)

    yield mesh_dev

    # Do NOT close — the manager owns the lifecycle.
    # Per-test cleanup is handled by _reset_mesh_device_state below.


# ---------------------------------------------------------------------------
# Per-test mesh device state reset (autouse)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_mesh_device_state(request, _mesh_device_manager):
    """
    Reset mesh device state after each test so the shared device behaves
    cleanly for the next test.

    - Clears loaded sub-device managers (tests may load their own)
    - Resets sub-device stall groups
    - Resets cache_entries_counter delta
    - Does NOT call clear_program_cache() — this breaks ethernet CQ init
    """
    yield

    md = _mesh_device_manager.mesh_device
    if md is not None:
        try:
            md.clear_loaded_sub_device_manager()
        except Exception:
            pass
        try:
            md.reset_sub_device_stall_group()
        except Exception:
            pass
        try:
            md.cache_entries_counter.reset()
        except Exception:
            pass
        # Re-enable program cache in case a test called
        # disable_and_clear_program_cache().  This is safe — it does
        # NOT clear existing entries, it just ensures the cache is
        # active for the next test.
        try:
            md.enable_program_cache()
        except Exception:
            pass

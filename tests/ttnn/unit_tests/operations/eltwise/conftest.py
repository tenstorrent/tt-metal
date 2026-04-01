# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Module-scoped ``device`` fixture for eltwise unit tests.

Overrides the root function-scoped ``device`` fixture so that a single
device is opened once per test module and shared across all ~2800+
eltwise tests.  This eliminates thousands of redundant CreateDevice /
close_device round-trips in CI, cutting job wall-time significantly.

Tests marked with ``@pytest.mark.requires_fresh_device`` get a dedicated
per-test device with an empty program cache.  The shared device is
temporarily closed while the fresh device is in use.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
from loguru import logger
from tests.scripts.common import get_updated_device_params


# ---------------------------------------------------------------------------
# Root-conftest loader (for helpers defined only in the repo-root conftest)
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    for env in ("TT_METAL_HOME", "TT_METAL_ROOT"):
        v = os.environ.get(env)
        if v:
            return Path(v).resolve()
    return Path(__file__).resolve().parents[5]


_ROOT = _repo_root()
_ROOT_CONFTEST_MODULE = "tt_metal_root_conftest_for_ttnn_distributed"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _root_conftest():
    """Load repo-root conftest in a dedicated module name so helpers are available without clashing."""
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
        "requires_fresh_device: test needs a dedicated per-test device (empty program cache)",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device_id(config):
    """Return the device_id from CLI options, accounting for TG clusters."""
    device_id = config.getoption("device_id")
    if _rc.is_tg_cluster() and not device_id:
        device_id = _rc.first_available_tg_device()
    return device_id


def _open_device(device_id):
    """Open a device and attach a CacheEntriesCounter."""
    import ttnn

    from tests.tests_common.cache_entries_counter import CacheEntriesCounter

    updated_device_params = get_updated_device_params({})
    dev = ttnn.CreateDevice(device_id=device_id, **updated_device_params)
    ttnn.SetDefaultDevice(dev)
    dev.cache_entries_counter = CacheEntriesCounter(dev)
    return dev


def _close_device(dev):
    import ttnn

    ttnn.close_device(dev)


# ---------------------------------------------------------------------------
# Module-scoped device manager
# ---------------------------------------------------------------------------

class _DeviceManager:
    """
    Manages a module-scoped shared device that can be temporarily closed
    when a test needs a fresh device.

    The shared device is opened lazily on first ``get()`` call, avoiding
    unnecessary open/close cycles when all tests in a module use fresh
    devices (e.g. program-cache test modules).
    """

    def __init__(self, device_id):
        self.device = None
        self.device_id = device_id

    def get(self):
        """Return the shared device, opening it if necessary."""
        if self.device is None:
            self.device = _open_device(self.device_id)
            logger.info("Module-scoped eltwise device opened (device_id={})", self.device_id)
        return self.device

    def close(self):
        if self.device is not None:
            _close_device(self.device)
            logger.info("Module-scoped eltwise device closed")
            self.device = None

    def suspend(self):
        """Temporarily close the shared device so a fresh device can use the hardware."""
        if self.device is not None:
            _close_device(self.device)
            logger.info("Module-scoped eltwise device suspended for fresh-device test")
            self.device = None

    def resume(self):
        """Reopen the shared device after a fresh-device test completes.

        Note: this is a no-op — the device will be lazily reopened on next
        ``get()`` call.  This avoids unnecessary reopen when the next test
        also needs a fresh device.
        """
        pass


@pytest.fixture(scope="module")
def _device_manager(request):
    """Module-scoped device manager that owns the shared device lifecycle."""
    import ttnn

    device_id = _resolve_device_id(request.config)
    mgr = _DeviceManager(device_id)
    request.node.pci_ids = [ttnn.GetPCIeDeviceID(device_id)]
    yield mgr
    mgr.close()


# ---------------------------------------------------------------------------
# Per-test device fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def device(request, _device_manager):
    """
    Per-test device fixture for eltwise tests.

    By default, returns the shared module-scoped device.  If the test is
    decorated with ``@pytest.mark.requires_fresh_device``, the shared
    device is temporarily closed and a brand-new device is opened for that
    single test, then closed — and the shared device is reopened.

    This is necessary for tests that depend on empty program-cache state
    (e.g. sub-device core-intersection checks, absolute cache-count
    assertions) because ``clear_program_cache()`` cannot be used safely.
    """
    import ttnn

    if request.node.get_closest_marker("requires_fresh_device"):
        # Close shared device (if open) so the hardware is free
        _device_manager.suspend()

        device_id = _device_manager.device_id
        request.node.pci_ids = [ttnn.GetPCIeDeviceID(device_id)]
        fresh = _open_device(device_id)

        logger.info("Fresh per-test device opened for {}", request.node.name)
        yield fresh

        _close_device(fresh)
        logger.info("Fresh per-test device closed for {}", request.node.name)
        # Shared device will be lazily reopened on next get() call
    else:
        yield _device_manager.get()


# ---------------------------------------------------------------------------
# Per-test device-state reset (autouse)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_device_state(device):
    """
    Reset shared device state before each test so module-scoped device
    behaves like a fresh device for state-sensitive tests.

    Clears:
    - Loaded sub-device manager: tests that call load_sub_device_manager()
      without cleanup would otherwise leave the sub-device config active for
      the next test, causing 'num_intersections == num_cores' fatals.
    - cache_entries_counter: resets the delta counter so per-test cache-entry
      assertions (which measure new entries since reset, not absolute counts)
      start from zero.

    Note: program cache is intentionally NOT cleared here. Clearing it
    invalidates pre-compiled ethernet dispatch kernel binaries stored in
    command_queue_programs_, causing 'binary not found' fatals during CQ
    re-init in tests that use ethernet dispatch (e.g. test_ttnn_where_forge).
    Tests that check absolute program cache entry counts must instead use
    @pytest.mark.requires_fresh_device to get a dedicated device.
    """
    device.clear_loaded_sub_device_manager()
    device.cache_entries_counter.reset()
    yield

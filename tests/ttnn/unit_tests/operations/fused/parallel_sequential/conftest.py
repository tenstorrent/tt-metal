# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Module-scoped device fixture for parallel_sequential tests.

This subdirectory's tests are single-device with no ``device_params``
parametrization, so the simpler eltwise-style :class:`DeviceManager`
(module-scoped) is the right fit — there is no per-config keying to
do, and a module-scoped device gives the same wall-time win as the
session-scoped param-keyed pattern used elsewhere.

The ``demo/`` subdirectory has its own ``conftest.py`` that opts out of
collection unless explicitly targeted, so this module-scoped fixture
applies to ``test_parallel_sequential*.py`` and ``test_fusion_cache.py``
in this directory.

Tests marked with ``@pytest.mark.requires_fresh_device`` still get a
per-test fresh device; tests marked with ``@pytest.mark.manages_own_device``
suspend the shared device while they manage their own handle.

Note: a number of tests in ``test_parallel_sequential.py`` call
``device.disable_and_clear_program_cache()`` and re-enable it inside the
test body.  That is unaffected by the module-scoped fixture: the helper
:func:`reset_device_state` does NOT clear the program cache.
"""
import os
import sys
from pathlib import Path

import pytest

# Bootstrap repo root on sys.path so we can import the helper module.
_REPO_ROOT = Path(
    os.environ.get("TT_METAL_HOME") or os.environ.get("TT_METAL_ROOT") or Path(__file__).resolve().parents[6]
).resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.ttnn.conftest_helpers import (
    DeviceManager,
    register_device_markers,
    resolve_device_id,
    reset_device_state,
    _open_device,
    _close_device,
)


def pytest_configure(config):
    register_device_markers(config)


@pytest.fixture(scope="module")
def _device_manager(request):
    import ttnn

    device_id = resolve_device_id(request.config)
    mgr = DeviceManager(device_id, label="parallel_sequential")
    yield mgr
    mgr.close()


@pytest.fixture(scope="function")
def device(request, _device_manager):
    import ttnn

    if request.node.get_closest_marker("manages_own_device"):
        _device_manager.suspend()
        yield None
        return

    if request.node.get_closest_marker("requires_fresh_device"):
        _device_manager.suspend()
        device_id = _device_manager.device_id
        request.node.pci_ids = [ttnn.GetPCIeDeviceID(device_id)]
        fresh = _open_device(device_id)
        yield fresh
        _close_device(fresh)
        return

    request.node.pci_ids = [ttnn.GetPCIeDeviceID(_device_manager.device_id)]
    yield _device_manager.get()


@pytest.fixture(autouse=True)
def _reset_device_state_fixture(request, _device_manager):
    if request.node.get_closest_marker("manages_own_device"):
        yield
        return
    yield
    if _device_manager.device is not None:
        reset_device_state(_device_manager.device)

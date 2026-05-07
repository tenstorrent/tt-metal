# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Param-keyed session device fixture for data_movement tests.

Replaces the previous stub: this conftest installs a session-scoped
:class:`ParamKeyedDeviceManager` that auto-detects compatible runs across
files in this directory.  A handful of files (test_core.py,
test_embedding.py, test_non_zero_indices.py, test_tilize.py) parametrize
``device_params``; everything else uses the default ``{}`` config.  The
manager keeps the cached device when the key matches and reopens
automatically when it changes — adjacent tests with the same
``device_params`` no longer pay a CreateDevice/close round-trip per test.

A few tests in this directory request ``mesh_device`` rather than
``device``; those continue to flow through the root ``mesh_device``
fixture and are unaffected by this conftest.

Tests marked with ``@pytest.mark.requires_fresh_device`` still get a
per-test fresh device; tests marked with ``@pytest.mark.manages_own_device``
suspend the shared device while they manage their own handle.
"""
import os
import sys
from pathlib import Path

import pytest

# Bootstrap repo root on sys.path so we can import the helper module.
_REPO_ROOT = Path(
    os.environ.get("TT_METAL_HOME") or os.environ.get("TT_METAL_ROOT") or Path(__file__).resolve().parents[5]
).resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.ttnn.conftest_helpers import (
    ParamKeyedDeviceManager,
    register_device_markers,
    resolve_device_id,
    reset_device_state,
    sort_items_by_device_params,
    _open_device,
    _close_device,
)


def pytest_configure(config):
    register_device_markers(config)


@pytest.fixture(scope="session")
def _device_manager(request):
    import ttnn

    device_id = resolve_device_id(request.config)
    mgr = ParamKeyedDeviceManager(device_id, label="data_movement")
    yield mgr
    mgr.close()


@pytest.fixture(scope="function")
def device(request, _device_manager, device_params):
    import ttnn

    if request.node.get_closest_marker("manages_own_device"):
        _device_manager.suspend()
        yield None
        return

    if request.node.get_closest_marker("requires_fresh_device"):
        _device_manager.suspend()
        device_id = _device_manager.device_id
        request.node.pci_ids = [ttnn.GetPCIeDeviceID(device_id)]
        fresh = _open_device(device_id, device_params)
        yield fresh
        _close_device(fresh)
        return

    request.node.pci_ids = [ttnn.GetPCIeDeviceID(_device_manager.device_id)]
    yield _device_manager.get(device_params, request)


@pytest.fixture(autouse=True)
def _reset_device_state_fixture(request, _device_manager):
    if request.node.get_closest_marker("manages_own_device"):
        yield
        return
    yield
    if _device_manager.device is not None:
        reset_device_state(_device_manager.device)


def pytest_collection_modifyitems(config, items):
    sort_items_by_device_params(items)

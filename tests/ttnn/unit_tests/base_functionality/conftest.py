# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Param-keyed session device fixture for base_functionality tests.

This directory contains ~57 test files (~352 ``def test_*`` functions,
roughly 420 collected items after parametrization).  Many of those files
parametrize ``device_params`` indirectly (e.g. ``l1_small_size``,
``trace_region_size``, ``num_command_queues``, ``worker_l1_size``,
``dispatch_core_axis``, ``fabric_config``); the previous stub bailed
out on that case.  A session-scoped :class:`ParamKeyedDeviceManager`
auto-detects compatible runs across files: tests that share
``device_params`` reuse the cached device without any open/close, and
the device is transparently reopened when the params key changes.

No file in this directory currently uses
``@pytest.mark.use_module_device``, so this conftest does not need to
preserve that marker's semantics; the param-keyed manager already
subsumes the use_module_device benefit and extends it across files.

Tests marked ``@pytest.mark.requires_fresh_device`` still get a per-test
fresh device; tests marked ``@pytest.mark.manages_own_device`` suspend
the shared device while they manage their own handle.

Multi-device tests in this directory (e.g. ``test_multi_device*``,
``test_multi_host_clusters``, ``test_global_semaphore``,
``test_global_circular_buffer``, ``test_sub_device``,
``test_copy_point_to_point``, ``test_reshape``, ``test_narrow``,
``test_python_tracer``, ``test_to_layout``, ``test_graph_report``)
consume the ``mesh_device`` fixture, not ``device``.  They continue to
use the root conftest's ``mesh_device`` fixture (function-scoped) and
fall outside this conftest's scope.
"""
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(
    os.environ.get("TT_METAL_HOME") or os.environ.get("TT_METAL_ROOT") or Path(__file__).resolve().parents[4]
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
    mgr = ParamKeyedDeviceManager(device_id, label="base_functionality")
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

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Session-scoped mesh fixtures for BH multi-card box all_post_commit CCL.

Mirrors tests/nightly/t3000/ccl/conftest.py: a single mesh is opened
once per (mesh_shape, device_params) configuration and reused across
adjacent tests that share that config.  Configs change → mesh is closed
and reopened.

Tests in this directory consume ``bh_1d_mesh_device`` and
``bh_2d_mesh_device`` (NOT the generic ``mesh_device``).  Both are
overridden here to route through the shared session-scoped manager,
following the same shape logic as the root-conftest fixtures
(``bh_1d_mesh_device`` → ``(n, 1)``; ``bh_2d_mesh_device`` →
``(4, 2)`` for 8 devices, ``(4, 8)`` for 32, else ``(n, 1)``).
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(
    os.environ.get("TT_METAL_HOME") or os.environ.get("TT_METAL_ROOT") or Path(__file__).resolve().parents[8]
).resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.ttnn.conftest_helpers import (
    ParamKeyedMeshDeviceManager,
    register_mesh_device_markers,
    reset_mesh_device_state,
    sort_items_by_device_params,
)


_SUPPORTED_DEVICE_COUNTS = (1, 2, 4, 8, 32)


def _bh_1d_shape(num_devices: int) -> tuple:
    return (num_devices, 1)


def _bh_2d_shape(num_devices: int) -> tuple:
    if num_devices == 8:
        return (4, 2)
    if num_devices == 32:
        return (4, 8)
    return (num_devices, 1)


def _synthetic_request(request, shape):
    return SimpleNamespace(param=shape, node=request.node, config=request.config)


def pytest_configure(config):
    register_mesh_device_markers(config)


@pytest.fixture(scope="session")
def _mesh_device_manager():
    mgr = ParamKeyedMeshDeviceManager(label="bh_box_all_post_commit")
    yield mgr
    mgr.close()


@pytest.fixture(scope="function")
def mesh_device(request, silicon_arch_name, device_params, _mesh_device_manager):
    import ttnn

    request.node.pci_ids = ttnn.get_pcie_device_ids()
    yield _mesh_device_manager.get(device_params, request)


@pytest.fixture(scope="function")
def bh_1d_mesh_device(request, silicon_arch_name, silicon_arch_blackhole, device_params, _mesh_device_manager):
    import ttnn

    if ttnn.get_num_devices() not in _SUPPORTED_DEVICE_COUNTS:
        pytest.skip()
    request.node.pci_ids = ttnn.get_pcie_device_ids()
    shape = _bh_1d_shape(ttnn.get_num_devices())
    yield _mesh_device_manager.get(device_params, _synthetic_request(request, shape))


@pytest.fixture(scope="function")
def bh_2d_mesh_device(request, silicon_arch_name, silicon_arch_blackhole, device_params, _mesh_device_manager):
    import ttnn

    if ttnn.get_num_devices() not in _SUPPORTED_DEVICE_COUNTS:
        pytest.skip()
    request.node.pci_ids = ttnn.get_pcie_device_ids()
    shape = _bh_2d_shape(ttnn.get_num_devices())
    yield _mesh_device_manager.get(device_params, _synthetic_request(request, shape))


@pytest.fixture(autouse=True)
def _reset_mesh_device_state_fixture(_mesh_device_manager):
    yield
    md = _mesh_device_manager.mesh_device
    if md is not None:
        reset_mesh_device_state(md)


def pytest_collection_modifyitems(config, items):
    sort_items_by_device_params(items)

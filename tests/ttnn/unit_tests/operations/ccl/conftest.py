# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Session-scoped ``mesh_device`` for single-host CCL.

Mirrors tests/nightly/t3000/ccl/conftest.py: a single mesh is opened
once per (mesh_shape, device_params) configuration and reused across
adjacent tests that share that config.  Configs change → mesh is closed
and reopened.

Scope: applies to tests/ttnn/unit_tests/operations/ccl/ and any non-BH
subdirectory.  The blackhole_CI/ subtrees override this conftest with
their own per-directory conftests because their tests consume
``bh_1d_mesh_device`` / ``bh_2d_mesh_device`` rather than ``mesh_device``.
"""
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(
    os.environ.get("TT_METAL_HOME") or os.environ.get("TT_METAL_ROOT") or Path(__file__).resolve().parents[5]
).resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.ttnn.conftest_helpers import (
    ParamKeyedMeshDeviceManager,
    register_mesh_device_markers,
    reset_mesh_device_state,
    sort_items_by_device_params,
)


def pytest_configure(config):
    register_mesh_device_markers(config)


@pytest.fixture(scope="session")
def _mesh_device_manager():
    mgr = ParamKeyedMeshDeviceManager(label="ccl_single_host")
    yield mgr
    mgr.close()


@pytest.fixture(scope="function")
def mesh_device(request, silicon_arch_name, device_params, _mesh_device_manager):
    import ttnn

    request.node.pci_ids = ttnn.get_pcie_device_ids()
    yield _mesh_device_manager.get(device_params, request)


@pytest.fixture(autouse=True)
def _reset_mesh_device_state_fixture(_mesh_device_manager):
    yield
    md = _mesh_device_manager.mesh_device
    if md is not None:
        reset_mesh_device_state(md)


def pytest_collection_modifyitems(config, items):
    sort_items_by_device_params(items)

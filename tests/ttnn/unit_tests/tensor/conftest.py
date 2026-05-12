# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Module-scoped ``device`` fixture for tensor unit tests.

Overrides the root function-scoped ``device`` fixture so that a single
device is opened once per test module and shared across all tensor
tests.  This eliminates redundant CreateDevice / close_device round-trips
in CI, cutting job wall-time significantly.

Note: this directory is collected as part of the "core ttnn unit test group"
alongside ``base_functionality/`` and ``benchmarks/``.  Tests in those
sibling directories continue to use the root function-scoped fixture;
only tests under ``tensor/`` benefit from this session-level sharing.
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
    return Path(__file__).resolve().parents[4]


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
# Session-scoped device fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device(request):
    """
    Open a single device per test module for tensor tests.

    This shadows the root function-scoped ``device`` fixture for every
    test collected under ``tests/ttnn/unit_tests/tensor/``.
    """
    import ttnn

    from tests.tests_common.cache_entries_counter import CacheEntriesCounter

    device_id = request.config.getoption("device_id")

    # On TG clusters, target the first user-exposed device, not device 0.
    if _rc.is_tg_cluster() and not device_id:
        device_id = _rc.first_available_tg_device()

    request.node.pci_ids = [ttnn.GetPCIeDeviceID(device_id)]

    updated_device_params = get_updated_device_params({})
    device = ttnn.CreateDevice(device_id=device_id, **updated_device_params)
    ttnn.SetDefaultDevice(device)

    device.cache_entries_counter = CacheEntriesCounter(device)

    logger.info("Module-scoped tensor device opened (device_id={})", device_id)
    yield device

    ttnn.close_device(device)
    logger.info("Module-scoped tensor device closed")

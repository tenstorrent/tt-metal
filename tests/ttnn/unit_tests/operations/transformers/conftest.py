# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Directory-local fixtures for transformers op tests.

Session-scoped device fixture not applied: tests in this group use
@pytest.mark.parametrize("device_params", ...) which conflicts with a
shared session device. Refactor those tests first.

Do not define ``mesh_device`` here: it would shadow the repo-root fixture and break
indirect parametrization (e.g. ``test_prefetcher_TG.py`` (2,2), ``test_paged_cache_mask.py`` (1,2)).

Module-scoped ``prefetcher_multi_device_mesh`` opens the mesh once per test module (per xdist worker).
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
from loguru import logger

from tests.scripts.common import get_updated_device_params


def _repo_root() -> Path:
    for env in ("TT_METAL_HOME", "TT_METAL_ROOT"):
        v = os.environ.get(env)
        if v:
            return Path(v).resolve()
    # tests/ttnn/unit_tests/operations/transformers/conftest.py -> parents[5] == repo root
    return Path(__file__).resolve().parents[5]


_ROOT = _repo_root()
# Same module name as tests/ttnn/distributed/conftest.py so root conftest loads once per process.
_ROOT_CONTEST_MODULE = "tt_metal_root_conftest_for_ttnn_distributed"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _root_conftest():
    """Load repo-root conftest in a dedicated module name so helpers are available without clashing."""
    if _ROOT_CONTEST_MODULE in sys.modules:
        return sys.modules[_ROOT_CONTEST_MODULE]
    path = _ROOT / "conftest.py"
    spec = importlib.util.spec_from_file_location(_ROOT_CONTEST_MODULE, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_ROOT_CONTEST_MODULE] = mod
    spec.loader.exec_module(mod)
    return mod


_rc = _root_conftest()


@pytest.fixture(scope="module")
def prefetcher_multi_device_mesh(request):
    """Full-system mesh with fixed trace region for multi-device prefetcher tests.

    Scope is module so all parametrized cases of ``test_run_prefetcher_post_commit_multi_device`` share one
    open/close. Under pytest-xdist each worker has its own module instance (one mesh per worker per module).
    """
    import ttnn

    from tests.tests_common.cache_entries_counter import CacheEntriesCounter

    rc = _rc
    request.node.pci_ids = ttnn.get_pcie_device_ids()

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
        mesh_shape = ttnn.MeshShape(*grid_dims)
    else:
        if not ttnn.using_distributed_env() and param > ttnn.get_num_devices():
            pytest.skip("Requested more devices than available. Test not applicable for machine")
        mesh_shape = ttnn.MeshShape(1, param)

    device_params = {"trace_region_size": 23887872}
    updated_device_params = get_updated_device_params(device_params)
    fabric_config = updated_device_params.pop("fabric_config", None)
    fabric_tensix_config = updated_device_params.pop("fabric_tensix_config", None)
    reliability_mode = updated_device_params.pop("reliability_mode", None)
    fabric_manager = updated_device_params.pop("fabric_manager", None)
    fabric_router_config = updated_device_params.pop("fabric_router_config", None)
    rc.set_fabric(fabric_config, reliability_mode, fabric_tensix_config, fabric_manager, fabric_router_config)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)

    mesh_device.cache_entries_counter = CacheEntriesCounter(mesh_device)

    logger.debug(f"prefetcher_multi_device_mesh: {mesh_device.get_num_devices()} devices")
    yield mesh_device

    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)

    ttnn.close_mesh_device(mesh_device)
    rc.reset_fabric(fabric_config)
    del mesh_device

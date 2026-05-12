# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Session-scoped mesh_device for tests under this directory (overrides root function-scoped fixture)."""

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
    return Path(__file__).resolve().parents[3]


_ROOT = _repo_root()
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


@pytest.fixture(scope="session")
def mesh_device(request):
    import ttnn

    from models.tt_transformers.demo.trace_region_config import get_supported_trace_region_size
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

    device_params = {}
    # TG (8×4) and other large multi-device meshes require COL dispatch for ethernet dispatch.
    # Apply this unconditionally for all tests under tests/ttnn/distributed/ rather than
    # requiring each test to inject it via @pytest.mark.parametrize("device_params", ..., indirect=True),
    # which does not work with a session-scoped mesh_device fixture.
    device_params.setdefault("dispatch_core_axis", ttnn.DispatchCoreAxis.COL)
    override_trace_region_size = get_supported_trace_region_size(request, param)
    if override_trace_region_size:
        device_params["trace_region_size"] = override_trace_region_size
        logger.info(f"Overriding trace region size to {override_trace_region_size}")

    updated_device_params = get_updated_device_params(device_params)
    fabric_config = updated_device_params.pop("fabric_config", None)
    fabric_tensix_config = updated_device_params.pop("fabric_tensix_config", None)
    reliability_mode = updated_device_params.pop("reliability_mode", None)
    fabric_manager = updated_device_params.pop("fabric_manager", None)
    fabric_router_config = updated_device_params.pop("fabric_router_config", None)
    rc.set_fabric(fabric_config, reliability_mode, fabric_tensix_config, fabric_manager, fabric_router_config)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)

    mesh_device.cache_entries_counter = CacheEntriesCounter(mesh_device)

    logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created")
    yield mesh_device

    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)

    ttnn.close_mesh_device(mesh_device)
    rc.reset_fabric(fabric_config)
    del mesh_device

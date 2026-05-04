# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
from loguru import logger

from models.demos.gpt_oss.tt.model_config import ModelArgs


def pytest_addoption(parser):
    parser.addoption("--skip-model-load", action="store_true", default=False, help="Skip loading the model state dict")


@pytest.fixture(scope="session")
def state_dict(request):
    load_model = not request.config.getoption("--skip-model-load")
    model_path = os.getenv("HF_MODEL", None)
    if model_path is None or not load_model:
        return {}
    raw = model_path.strip()
    expanded = os.path.expanduser(raw)
    if (os.path.isabs(expanded) or raw.startswith("~/")) and not os.path.isdir(expanded):
        pytest.skip(
            f"HF_MODEL must be an existing local checkpoint directory; got {model_path!r} "
            f"(resolved {expanded!r}). Do not use the docs placeholder /path/to/.... "
            f"Example: export HF_MODEL=/local/ttuser/apande/models/gpt-oss-120b"
        )
    return ModelArgs.load_state_dict(expanded, dummy_weights=False)


@pytest.fixture
def test_thresholds(request):
    with open("models/demos/gpt_oss/unit_test_thresholds.json", "r") as f:
        thresholds = json.load(f)
    return thresholds


_SESSION_MESH_ENABLED_ENV = "GPT_OSS_SESSION_MESH"


def _session_mesh_enabled() -> bool:
    return os.getenv(_SESSION_MESH_ENABLED_ENV, "").lower() in ("1", "true", "yes", "y")


class _SessionMeshCache:
    def __init__(self):
        self._entries: dict = {}

    def key(self, param, device_params):
        pass

        if isinstance(param, tuple):
            shape = tuple(param)
        else:
            shape = (1, int(param))
        fabric = device_params.get("fabric_config", None)
        fabric_id = getattr(fabric, "name", str(fabric)) if fabric is not None else None
        trace = device_params.get("trace_region_size", None)
        return (shape, fabric_id, trace)

    def get(self, key):
        return self._entries.get(key)

    def put(self, key, value):
        self._entries[key] = value

    def close_all(self):
        import ttnn
        from conftest import reset_fabric

        for (shape, _fabric_id, _trace), entry in self._entries.items():
            mesh = entry.get("mesh")
            fabric_config = entry.get("fabric_config")
            if mesh is None:
                continue
            try:
                for submesh in mesh.get_submeshes():
                    ttnn.close_mesh_device(submesh)
                ttnn.close_mesh_device(mesh)
            except Exception as e:
                logger.warning(f"Failed to close session mesh {shape}: {e}")
            try:
                reset_fabric(fabric_config)
            except Exception as e:
                logger.warning(f"Failed to reset fabric for session mesh {shape}: {e}")
        self._entries.clear()


@pytest.fixture(scope="session")
def _gpt_oss_session_mesh_cache():
    cache = _SessionMeshCache()
    try:
        yield cache
    finally:
        cache.close_all()


@pytest.fixture(scope="function")
def mesh_device(request, silicon_arch_name, device_params, _gpt_oss_session_mesh_cache):
    import ttnn
    from conftest import get_supported_trace_region_size, get_updated_device_params, reset_fabric, set_fabric

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

    override_trace_region_size = get_supported_trace_region_size(request, param)
    if override_trace_region_size:
        device_params["trace_region_size"] = override_trace_region_size
        logger.info(f"Overriding trace region size to {override_trace_region_size}")

    updated_device_params = get_updated_device_params(dict(device_params))
    fabric_config = updated_device_params.pop("fabric_config", None)
    fabric_tensix_config = updated_device_params.pop("fabric_tensix_config", None)
    reliability_mode = updated_device_params.pop("reliability_mode", None)
    fabric_manager = updated_device_params.pop("fabric_manager", None)
    fabric_router_config = updated_device_params.pop("fabric_router_config", None)

    session_mode = _session_mesh_enabled()
    cache_key = _gpt_oss_session_mesh_cache.key(param, dict(device_params))
    cached_entry = _gpt_oss_session_mesh_cache.get(cache_key) if session_mode else None

    if cached_entry is not None:
        logger.info(f"Reusing session-scoped mesh {cache_key}")
        mesh = cached_entry["mesh"]
        mesh.cache_entries_counter.reset() if hasattr(mesh.cache_entries_counter, "reset") else None
        yield mesh
        return

    set_fabric(fabric_config, reliability_mode, fabric_tensix_config, fabric_manager, fabric_router_config)
    mesh = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)

    from tests.tests_common.cache_entries_counter import CacheEntriesCounter

    mesh.cache_entries_counter = CacheEntriesCounter(mesh)

    logger.debug(f"multidevice with {mesh.get_num_devices()} devices is created")

    if session_mode:
        _gpt_oss_session_mesh_cache.put(
            cache_key,
            {"mesh": mesh, "fabric_config": fabric_config},
        )
        yield mesh
        return

    try:
        yield mesh
    finally:
        for submesh in mesh.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh)
        reset_fabric(fabric_config)
        del mesh

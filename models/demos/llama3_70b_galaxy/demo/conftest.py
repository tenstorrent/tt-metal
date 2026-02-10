# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from pathlib import Path
from loguru import logger


# Default device params matching text_demo.py parametrize (single value)
TEXT_DEMO_DEVICE_PARAMS = {
    "trace_region_size": 184915840,
    "num_command_queues": 1,
    "dispatch_core_axis": None,  # set below with ttnn
    "worker_l1_size": 1345000,
    "fabric_config": True,
}


@pytest.fixture(scope="session")
def mesh_device(silicon_arch_name):
    """
    Session-scoped mesh device so the model can be loaded once and reused across
    text_demo tests. Uses the same (8, 4) shape and device_params as text_demo parametrize.
    """
    import ttnn
    from tests.scripts.common import get_updated_device_params

    # Resolve fabric_config and dispatch_core_axis like text_demo's device_params fixture
    params = TEXT_DEMO_DEVICE_PARAMS.copy()
    if ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY:
        params["fabric_config"] = ttnn.FabricConfig.FABRIC_1D_RING
    elif ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.TG:
        params["fabric_config"] = ttnn.FabricConfig.FABRIC_1D
    else:
        params["fabric_config"] = None
    params["dispatch_core_axis"] = ttnn.DispatchCoreAxis.COL

    # Use root conftest helpers for device creation/teardown (repo root = tt-metal)
    import importlib.util

    root_conftest_path = Path(__file__).resolve().parents[4] / "conftest.py"
    spec = importlib.util.spec_from_file_location("root_conftest", root_conftest_path)
    root_conftest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(root_conftest)
    set_fabric = root_conftest.set_fabric
    reset_fabric = root_conftest.reset_fabric

    grid_dims = (8, 4)
    num_devices_requested = grid_dims[0] * grid_dims[1]
    if not ttnn.using_distributed_env() and num_devices_requested > ttnn.get_num_devices():
        pytest.skip("Requested more devices than available. Test not applicable for machine")
    mesh_shape = ttnn.MeshShape(*grid_dims)

    updated_device_params = get_updated_device_params(params)
    fabric_config = updated_device_params.pop("fabric_config", None)
    fabric_tensix_config = updated_device_params.pop("fabric_tensix_config", None)
    reliability_mode = updated_device_params.pop("reliability_mode", None)
    fabric_manager = updated_device_params.pop("fabric_manager", None)
    fabric_router_config = updated_device_params.pop("fabric_router_config", None)
    set_fabric(fabric_config, reliability_mode, fabric_tensix_config, fabric_manager, fabric_router_config)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)

    logger.debug(f"session mesh_device with {mesh_device.get_num_devices()} devices created")
    yield mesh_device

    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh_device)
    reset_fabric(fabric_config)
    del mesh_device


def pytest_collection_modifyitems(config, items):
    """Apply longer timeout to text_demo test in this directory (device init + model load + prefill can exceed default 300s)."""
    for item in items:
        if "text_demo.py::test_demo_text" in item.nodeid:
            item.add_marker(pytest.mark.timeout(1500))


# These inputs override the default inputs used by simple_text_demo.py. Check the main demo to see the default values.
def pytest_addoption(parser):
    parser.addoption("--input_prompts", action="store", help="input prompts json file")
    parser.addoption("--instruct", action="store", type=int, help="Use instruct weights")
    parser.addoption("--repeat_batches", action="store", type=int, help="Number of consecutive batches of users to run")
    parser.addoption("--max_seq_len", action="store", type=int, help="Maximum context length supported by the model")
    parser.addoption("--batch_size", action="store", type=int, help="Number of users in a batch ")
    parser.addoption(
        "--max_generated_tokens", action="store", type=int, help="Maximum number of tokens to generate for each user"
    )
    parser.addoption(
        "--paged_attention", action="store", type=bool, help="Whether to use paged attention or default attention"
    )
    parser.addoption("--page_params", action="store", type=dict, help="Page parameters for paged attention")
    parser.addoption("--sampling_params", action="store", type=dict, help="Sampling parameters for decoding")
    parser.addoption(
        "--stop_at_eos", action="store", type=int, help="Whether to stop decoding when the model generates an EoS token"
    )
    parser.addoption(
        "--disable_pf_perf_mode", action="store_true", default=False, help="Enable performance mode for prefetcher"
    )
    parser.addoption(
        "--print_outputs",
        action="store",
        default=False,
        type=bool,
        help="Whether to print token output every decode iteration",
    )
    parser.addoption(
        "--prefill_profile",
        action="store",
        default=False,
        type=bool,
        help="Whether to enable prefill profile mode",
    )

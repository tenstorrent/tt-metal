# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from models.utility_functions import skip_for_grayskull
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

from tests.ttnn.unit_tests.operations.ccl.test_new_all_reduce import (
    SUB_DEVICE_CRS,
    QKV_CRS,
    RING_CRS,
    FF1_CRS,
    FF1_CRS_RS_OUT,
    NORM_CRS,
)

from tests.ttnn.unit_tests.operations.ccl.test_llama_reduce_scatter_async_TG import run_reduce_scatter_test

from conftest import is_6u


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 300000,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_links", [3, 4])
@pytest.mark.parametrize("topology", [ttnn.Topology.Ring])
def test_fabric_reduce_scatter_tg_trace_6u(mesh_device, trace_mode, num_links, topology):
    # Only run these tests on unharvested TG
    device_grid = (mesh_device.compute_with_storage_grid_size().x, mesh_device.compute_with_storage_grid_size().y)
    if device_grid != (7, 10):
        pytest.skip("Not a Galaxy!")

    if not is_6u():
        pytest.skip("Not 6U!")

    dim = 3
    shard_height = 32
    shard_width = 160
    num_devices_scatter = 4
    num_devices_fracture = 8
    num_cores = 24
    num_iters = 75
    warmup_iters = 10
    trace_mode = trace_mode
    num_links = 4

    run_reduce_scatter_test(
        mesh_device,
        dim,
        shard_height,
        shard_width,
        num_devices_scatter,
        num_devices_fracture,
        num_cores,
        num_iters,
        warmup_iters,
        trace_mode,
        num_links=num_links,
        scheme="random",
        topology=topology,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_links", [3, 4])
@pytest.mark.parametrize("topology", [ttnn.Topology.Ring, ttnn.Topology.Linear])
def test_fabric_reduce_scatter_tg_no_trace_6u(mesh_device, trace_mode, num_links, topology):
    # Only run these tests on unharvested TG
    device_grid = (mesh_device.compute_with_storage_grid_size().x, mesh_device.compute_with_storage_grid_size().y)
    if device_grid != (7, 10):
        pytest.skip("Not a Galaxy!")

    if not is_6u():
        pytest.skip("Not 6U!")

    dim = 3
    shard_height = 32
    shard_width = 160
    num_devices_scatter = 4
    num_devices_fracture = 8
    num_cores = 24
    num_iters = 30
    warmup_iters = 0
    trace_mode = trace_mode
    num_links = 4

    run_reduce_scatter_test(
        mesh_device,
        dim,
        shard_height,
        shard_width,
        num_devices_scatter,
        num_devices_fracture,
        num_cores,
        num_iters,
        warmup_iters,
        trace_mode,
        num_links=num_links,
        scheme="random",
        topology=topology,
    )

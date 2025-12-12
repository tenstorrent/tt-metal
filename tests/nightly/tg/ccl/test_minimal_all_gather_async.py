# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.nightly.t3000.ccl.test_minimal_all_gather_async import run_all_gather_impl
from models.common.utility_functions import skip_for_blackhole, skip_for_wormhole_b0
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc


@pytest.mark.parametrize("num_links", [2], ids=["2links"])
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, ag_input_dtype",
    [
        (4, [1, 1, 9472, 5120], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, [1, 10, 75776, 128], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
    ids=[
        "wan_ag_1",
        "wan_ag_2",
    ],
)
@pytest.mark.parametrize("num_iters, enable_trace", [(4, True), (1, False)], ids=["perf", "check"])
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
    ids=["DRAM_memconfig"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        # @Justin: please change to neighbor exchange config
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
@pytest.mark.parametrize(
    "all_gather_function",
    [ttnn.experimental.all_gather_async],
    ids=[
        "normal",
    ],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_all_gather_async(
    mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    all_gather_topology,
    num_iters,
    all_gather_function,
):
    if num_devices < 8:
        submesh_shape = (1, num_devices)
        cluster_axis = 1
    else:
        submesh_shape = (num_devices, 1)
        cluster_axis = 0
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(submesh_shape))
    run_all_gather_impl(
        submesh_device,
        num_devices,
        ag_output_shape,
        dim,
        num_links,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
        all_gather_function=all_gather_function,
    )
    ttnn.ReadDeviceProfiler(submesh_device)

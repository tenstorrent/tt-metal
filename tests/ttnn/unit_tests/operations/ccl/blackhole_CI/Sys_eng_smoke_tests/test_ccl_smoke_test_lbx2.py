# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.nightly.t3000.ccl.test_minimal_all_gather_async import run_all_gather_impl
from models.common.utility_functions import skip_for_wormhole_b0, run_for_n_dev
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test
from conftest import get_updated_device_params, set_fabric, reset_fabric


# Test for 1x16 mesh (16 devices in a row)
@skip_for_wormhole_b0()
@pytest.mark.parametrize("num_links", [2])
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, all_gather_topology, cluster_axis",
    [
        (16, [1, 1, 12000, 32768], 3, ttnn.TILE_LAYOUT, ttnn.Topology.Linear, 1),
    ],
    ids=[
        "1x16_row_test",
    ],
)
@pytest.mark.parametrize(
    "ag_input_dtype",
    [
        ttnn.bfloat16,
    ],
    ids=[
        "bfloat16",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
    ],
    ids=[
        "DRAM_ONLY",
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (False, 1),
    ],
    ids=["non-trace"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 90112}),
    ],
    indirect=["device_params"],
    ids=["fabric"],
)
@pytest.mark.parametrize("chunks_per_sync", [20])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [2])
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 16), id="1x16_grid")], indirect=True)
def test_ccl_ddr_smoke_test_1x16(
    mesh_device,
    num_devices,
    ag_output_shape,
    cluster_axis,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    all_gather_topology,
    num_iters,
    chunks_per_sync,
    num_workers_per_link,
    num_buffers_per_channel,
):
    validate_test(num_devices, all_gather_topology, mesh_device.shape, cluster_axis)
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))
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
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
        allowed_pcc=0.9999,
    )
    ttnn.ReadDeviceProfiler(submesh_device)

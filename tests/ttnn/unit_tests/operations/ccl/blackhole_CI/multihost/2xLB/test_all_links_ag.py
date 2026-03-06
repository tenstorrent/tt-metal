# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from tests.nightly.t3000.ccl.test_minimal_reduce_scatter_async import run_reduce_scatter_impl
import ttnn

from tests.nightly.t3000.ccl.test_minimal_all_gather_async import run_all_gather_impl
from models.common.utility_functions import skip_for_wormhole_b0, run_for_n_dev
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test


# P300 with 2 harvested columns so 110 cores are available.
# Test utilizes 1'478'492.16 bytes per core to nearly maximize 1.5MB size
@skip_for_wormhole_b0()
@pytest.mark.parametrize("num_links", [2])
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, all_gather_topology, ag_input_dtype",
    [
        (4, [1, 1, 256, 256], 3, ttnn.TILE_LAYOUT, ttnn.Topology.Linear, ttnn.bfloat16),
    ],
    ids=["horizontal_test_bf16", "vertical_test_u32", "vertical_test_bf8"],
)
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
    ],
    ids=[
        "L1_ONLY",
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 3),
    ],
    ids=["non-trace"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}),
    ],
    indirect=["device_params"],
    ids=["fabric"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((4, 4), id="4x4_grid")], indirect=True)
@pytest.mark.parametrize("chunks_per_sync", [20])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [2])
def test_all_links_ag(
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
    if cluster_axis == 0:
        print(f"Testing horizontal all-gather with {num_devices} devices and {num_links} links")
    else:
        print(f"Testing vertical all-gather with {num_devices} devices and {num_links} links")
    for i in range(mesh_device.shape[(cluster_axis - 1) % 2]):
        if cluster_axis == 0:
            print(f"Validating row {i} of {mesh_device.shape}")
        else:
            print(f"Validating column {i} of {mesh_device.shape}")
        if cluster_axis == 0:
            submesh_device = mesh_device.create_submesh(
                ttnn.MeshShape((num_devices, 1)), offset=ttnn.MeshCoordinate(0, i)
            )
        else:
            submesh_device = mesh_device.create_submesh(
                ttnn.MeshShape((1, num_devices)), offset=ttnn.MeshCoordinate(i, 0)
            )
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
            num_l1_banks=110,
        )
    ttnn.ReadDeviceProfiler(submesh_device)

@pytest.mark.parametrize("num_links", [2], ids=["2_links"])
@pytest.mark.parametrize(
    "num_devices, rs_input_shape, dim, layout",
    [
        (4, [1, 1, 128, 256], 3, ttnn.TILE_LAYOUT),
    ],
    ids=["4_device"],
)
@pytest.mark.parametrize(
    "rs_input_dtype",
    [
        ttnn.bfloat16,
    ],
    ids=[
        "float_16",
    ],
)
@pytest.mark.parametrize("cluster_axis", [0, 1])

@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
    ],
    ids=["dram_only", "l1_only"],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 3),
    ],
    ids=[
        "trace",
    ],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=[
        "fabric_linear",
    ],
)
@pytest.mark.parametrize("chunks_per_sync", [2])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [8])
@pytest.mark.parametrize("mesh_device", [pytest.param((4, 4), id="4x4_grid")], indirect=True)

def test_rs(
    mesh_device,
    num_devices,
    num_links,
    rs_input_shape,
    dim,
    layout,
    rs_input_dtype,
    mem_config_input,
    mem_config_rs,
    enable_trace,
    num_iters,
    rs_topology,
    chunks_per_sync,
    num_workers_per_link,
    cluster_axis,
    num_buffers_per_channel,
):
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    run_reduce_scatter_impl(
        submesh_device,
        num_devices,
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        layout,
        mem_config_input,
        mem_config_rs,
        rs_topology=rs_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
    )
    ttnn.ReadDeviceProfiler(submesh_device)
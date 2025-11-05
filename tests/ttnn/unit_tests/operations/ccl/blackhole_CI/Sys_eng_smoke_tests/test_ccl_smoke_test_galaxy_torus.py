# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.nightly.t3000.ccl.test_minimal_all_gather_async import run_all_gather_impl
from models.common.utility_functions import skip_for_wormhole_b0
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test


# Test uses 3.932GB of space per device to nearly fill the dram
@skip_for_wormhole_b0()
@pytest.mark.parametrize("num_links", [2])  # Check over all four links
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, all_gather_topology, cluster_axis",
    [
        (4, [1, 1, 24000, 32768], 3, ttnn.TILE_LAYOUT, ttnn.Topology.Ring, 0),
        (8, [1, 1, 20000, 32768], 3, ttnn.TILE_LAYOUT, ttnn.Topology.Ring, 1),
    ],
    ids=[
        "row_test",
        "column_test",
    ],
)
@pytest.mark.parametrize(
    "ag_input_dtype",
    [
        ttnn.bfloat16,
        ttnn.uint32,
        ttnn.bfloat8_b,
    ],
    ids=[
        "float_16",
        "uint_32",
        "bfloat_8",
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
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}),
    ],
    indirect=["device_params"],
    ids=["fabric"],
)
@pytest.mark.parametrize("chunks_per_sync", [20])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [2])
def test_ccl_ddr_smoke_test(
    bh_2d_mesh_device,
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
    validate_test(num_devices, all_gather_topology, bh_2d_mesh_device.shape, cluster_axis)
    # Check all the rows and columns independently within the device
    if cluster_axis == 0:
        submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    else:
        submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))
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


# P300 with 2 harvested columns so 110 cores are available.
# Test utilizes 1'478'492.16 bytes per core to nearly maximize 1.5MB size
@skip_for_wormhole_b0()
@pytest.mark.parametrize("num_links", [2])
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, all_gather_topology, cluster_axis",
    [
        (4, [1, 1, 6016, 8192], 3, ttnn.TILE_LAYOUT, ttnn.Topology.Ring, 0),
        (8, [1, 1, 6016, 4096], 3, ttnn.TILE_LAYOUT, ttnn.Topology.Ring, 1),
    ],
    ids=["horizontal_test", "vertical_test"],
)
@pytest.mark.parametrize(
    "ag_input_dtype",
    [
        ttnn.bfloat16,
        ttnn.uint32,
        ttnn.bfloat8_b,
    ],
    ids=[
        "float_16",
        "uint_32",
        "bfloat_8",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
    ],
    ids=[
        "L1_TO_DRAM",
        "DRAM_TO_L1",
        "L1_ONLY",
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
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}),
    ],
    indirect=["device_params"],
    ids=["fabric"],
)
@pytest.mark.parametrize("chunks_per_sync", [20])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [2])
def test_ccl_other_smoke_test(
    bh_2d_mesh_device,
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
    validate_test(num_devices, all_gather_topology, bh_2d_mesh_device.shape, cluster_axis)
    for i in range(bh_2d_mesh_device.shape[(cluster_axis - 1) % 2]):
        if cluster_axis == 0:
            submesh_device = bh_2d_mesh_device.create_submesh(
                ttnn.MeshShape((num_devices, 1)), offset=ttnn.MeshCoordinate(0, i)
            )
        else:
            submesh_device = bh_2d_mesh_device.create_submesh(
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

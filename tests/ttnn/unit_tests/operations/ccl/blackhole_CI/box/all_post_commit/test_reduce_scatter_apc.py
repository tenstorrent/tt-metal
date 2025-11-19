# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from tests.nightly.t3000.ccl.test_minimal_reduce_scatter_async import run_reduce_scatter_impl
from models.common.utility_functions import (
    skip_for_wormhole_b0,
    skip_for_n_dev,
    skip_for_n_or_less_dev,
)
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test


@skip_for_wormhole_b0("This test is for blackhole")
@skip_for_n_dev(8)
@pytest.mark.parametrize("num_links", [2], ids=["2_links"])
@pytest.mark.parametrize(
    "num_devices, rs_input_shape, dim, layout",
    [
        (4, [1, 1, 128, 256], 3, ttnn.TILE_LAYOUT),
        (2, [1, 1, 64, 512], 3, ttnn.TILE_LAYOUT),
    ],
    ids=["4_device", "2_device"],
)
@pytest.mark.parametrize(
    "rs_input_dtype",
    [
        ttnn.bfloat16,
        # ttnn.uint32, #Bad PCC
        ttnn.bfloat8_b,
    ],
    ids=[
        "float_16",
        # "uint_32", #Bad PCC
        "bfloat_8",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
    ],
    ids=["dram_only", "l1_only"],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 10),
        (False, 10),
    ],
    ids=[
        "trace",
        "non-trace",
    ],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_2D_DYNAMIC, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=[
        "fabric_linear",
    ],
)
@pytest.mark.parametrize("chunks_per_sync", [2])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [8])
def test_rs_row_2d(
    bh_2d_mesh_device,
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
    num_buffers_per_channel,
):
    if bh_2d_mesh_device.shape[0] != 1 and bh_2d_mesh_device.shape[1] != 1:
        pytest.skip("2D dynamic requires one dimension to be 1")
    validate_test(num_devices, rs_topology, bh_2d_mesh_device.shape, 0)
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    cluster_axis = 0
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


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("num_links", [2], ids=["2_links"])
@pytest.mark.parametrize(
    "num_devices, rs_input_shape, dim, layout",
    [
        (2, [1, 1, 64, 512], 3, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "rs_input_dtype",
    [
        ttnn.bfloat16,
        # ttnn.uint32, #Bad PCC
        ttnn.bfloat8_b,
    ],
    ids=[
        "float_16",
        # "uint_32", #Bad PCC
        "bfloat_8",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
    ],
    ids=["dram_only", "l1_only"],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 10),
        (False, 10),
    ],
    ids=[
        "trace",
        "non-trace",
    ],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("chunks_per_sync", [2])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [8])
def test_rs_row_2D_nightly_linear(
    bh_1d_mesh_device,
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
    num_buffers_per_channel,
):
    validate_test(num_devices, rs_topology, bh_1d_mesh_device.shape, 0)
    submesh_device = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    cluster_axis = 0
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


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(3)
@pytest.mark.parametrize("num_links", [2], ids=["2_links"])
@pytest.mark.parametrize(
    "num_devices, rs_input_shape, dim, layout",
    [
        (4, [1, 1, 128, 256], 3, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "rs_input_dtype",
    [
        ttnn.bfloat16,
        # ttnn.uint32, #Bad PCC
        ttnn.bfloat8_b,
    ],
    ids=[
        "float_16",
        # "uint_32", #Bad PCC
        "bfloat_8",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
    ],
    ids=["dram_only", "l1_only"],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 10),
        (False, 10),
    ],
    ids=[
        "trace",
        "non-trace",
    ],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("chunks_per_sync", [2])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [8])
def test_rs_row_4D_nightly_linear(
    bh_1d_mesh_device,
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
    num_buffers_per_channel,
):
    validate_test(num_devices, rs_topology, bh_1d_mesh_device.shape, 0)
    submesh_device = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    cluster_axis = 0
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


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(2)
@pytest.mark.parametrize("num_links", [2], ids=["2_links"])
@pytest.mark.parametrize(
    "rs_input_shape, dim, layout",
    [
        ([1, 1, 128, 256], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 128, 512], 3, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "rs_input_dtype",
    [
        ttnn.bfloat16,
        # ttnn.uint32, #Bad PCC
        ttnn.bfloat8_b,
    ],
    ids=[
        "float_16",
        # "uint_32", #Bad PCC
        "bfloat_8",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
    ],
    ids=["dram_only", "l1_only"],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 10),
        (False, 10),
    ],
    ids=[
        "trace",
        "non-trace",
    ],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("chunks_per_sync", [2])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [8])
def test_rs_row_nightly_ring(
    bh_1d_mesh_device,
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
    num_buffers_per_channel,
):
    num_devices = bh_1d_mesh_device.shape[0]
    validate_test(num_devices, rs_topology, bh_1d_mesh_device.shape, 0)
    submesh_device = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    cluster_axis = 0
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


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(7)
@pytest.mark.parametrize("num_links", [2], ids=["2_links"])
@pytest.mark.parametrize(
    "num_devices, rs_input_shape, dim, layout",
    [
        (2, [1, 1, 64, 512], 3, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "rs_input_dtype",
    [
        ttnn.bfloat16,
        # ttnn.uint32, #Bad PCC
        ttnn.bfloat8_b,
    ],
    ids=[
        "float_16",
        # "uint_32", #Bad PCC
        "bfloat_8",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
    ],
    ids=["dram_only", "l1_only"],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 10),
        (False, 10),
    ],
    ids=[
        "trace",
        "non-trace",
    ],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("chunks_per_sync", [2])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [8])
def test_rs_row_2D_nightly_linear(
    bh_2d_mesh_device,
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
    num_buffers_per_channel,
):
    cluster_axis = 1
    validate_test(num_devices, rs_topology, bh_2d_mesh_device.shape, cluster_axis)
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))
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

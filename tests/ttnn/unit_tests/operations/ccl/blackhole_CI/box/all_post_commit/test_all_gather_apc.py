# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import ttnn

from tests.nightly.t3000.ccl.test_all_gather import run_all_gather_impl
from models.common.utility_functions import (
    skip_for_wormhole_b0,
    skip_for_n_or_less_dev,
)
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout",
    [
        (2, [1, 1, 128, 128], 3, ttnn.TILE_LAYOUT),
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
    ids=["trace", "non-trace"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "sub_core_grids",
    (
        # multiple disjoint cores
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
            ]
        ),
    ),
)
def test_all_gather_subcore_grid(
    bh_2d_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    num_iters,
    sub_core_grids,
):
    validate_test(num_devices, None, bh_2d_mesh_device.shape, 0)

    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    cluster_axis = 0
    run_all_gather_impl(
        submesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
        allowed_pcc=0.9999,
        sub_core_grids=sub_core_grids,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout",
    [
        (2, [1, 1, 128, 128], 3, ttnn.TILE_LAYOUT),
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
    ids=["trace", "non-trace"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
    ],
    indirect=True,
)
def test_all_gather_2D_line(
    bh_2d_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    num_iters,
):
    validate_test(num_devices, None, bh_2d_mesh_device.shape, 0)

    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    cluster_axis = 0
    run_all_gather_impl(
        submesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
        allowed_pcc=0.9999,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(2)
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout",
    [
        (4, [1, 1, 128, 2048], 3, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "ag_input_dtype",
    [
        ttnn.bfloat16,
    ],
    ids=[
        "float_16",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
    ],
    ids=["l1_only"],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 10),
        (False, 10),
    ],
    ids=["trace", "non-trace"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
    ],
    indirect=True,
    ids=["fabric_1d_linear"],
)
def test_all_gather_4D_line(
    bh_2d_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    num_iters,
):
    validate_test(num_devices, None, bh_2d_mesh_device.shape, 0)
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    cluster_axis = 0
    run_all_gather_impl(
        submesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
        allowed_pcc=0.9999,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(2)
@pytest.mark.parametrize(
    "ag_output_shape, dim, layout",
    [
        ([1, 1, 128, 2048], 3, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "ag_input_dtype",
    [
        ttnn.bfloat16,
    ],
    ids=[
        "float_16",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
    ],
    ids=["l1_only"],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 10),
        (False, 10),
    ],
    ids=["trace", "non-trace"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112},
    ],
    indirect=True,
)
def test_all_gather_ring(
    bh_1d_mesh_device,
    ag_output_shape,
    dim,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    num_iters,
):
    num_devices = bh_1d_mesh_device.shape[0]
    validate_test(num_devices, None, bh_1d_mesh_device.shape, 0)

    submesh_device = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    cluster_axis = 0
    run_all_gather_impl(
        submesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
        allowed_pcc=0.9999,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(7)
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout",
    [
        (2, [1, 1, 128, 128], 3, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "ag_input_dtype",
    [
        ttnn.bfloat16,
    ],
    ids=[
        "float_16",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
    ],
    ids=["l1_only"],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (False, 10),
    ],
    ids=["non-trace"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
    ],
    indirect=True,
)
def test_all_gather_8D_vertical(
    bh_2d_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    num_iters,
):
    validate_test(num_devices, None, bh_2d_mesh_device.shape, 1)
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))
    cluster_axis = 1
    run_all_gather_impl(
        submesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
        allowed_pcc=0.9999,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


@skip_for_wormhole_b0("This test is for blackhole")
@skip_for_n_or_less_dev(3)
@pytest.mark.parametrize(
    "num_devices, ag_output_shape,dim",
    [
        (4, [3, 1, 4096, 576], 3),
        (4, [3, 4, 4096, 144], 1),
    ],
    ids=["gemma dim 3", "gemma dim 1"],
)
@pytest.mark.parametrize(
    "layout",
    [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    ids=[
        "tile",
        "row major",
    ],
)
@pytest.mark.parametrize(
    "ag_input_dtype",
    [
        ttnn.bfloat16,
    ],
    ids=[
        "float_16",
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
    ids=["dram_only"],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (False, 10),
    ],
    ids=["non-trace"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 150000},
    ],
    indirect=True,
    ids=["fabric_linear"],
)
def test_all_gather_failing_shapes(
    bh_1d_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    num_iters,
):
    validate_test(num_devices, None, bh_1d_mesh_device.shape, 0)
    submesh_device = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    cluster_axis = 0
    run_all_gather_impl(
        submesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
        allowed_pcc=0.9999,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(2)
@pytest.mark.parametrize("num_devices", [4])
@pytest.mark.parametrize("num_links", [2], ids=["2_links"])
@pytest.mark.parametrize(
    "ag_output_shape, dim, layout",
    [
        ([1, 1, 128, 128], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 128, 2048], 3, ttnn.TILE_LAYOUT),
        ([3, 1, 4096, 576], 3, ttnn.ROW_MAJOR_LAYOUT),
    ],
    ids=["small", "large", "gemma_dim3"],
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
    ids=["dram_only"],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 10),
        (False, 10),
    ],
    ids=["trace", "non-trace"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 150000}, ttnn.Topology.Linear),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 150000}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_linear", "fabric_ring"],
)
@pytest.mark.parametrize(
    "sub_core_grids",
    (
        # multiple disjoint cores
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
            ]
        ),
    ),
)
def test_all_gather_experimental(
    bh_2d_mesh_device,
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
    sub_core_grids,
):
    validate_test(num_devices, all_gather_topology, bh_2d_mesh_device.shape, 0)
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    cluster_axis = 0

    # Composite path doesn't support sub core grids
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        sub_core_grids = None

    run_all_gather_impl(
        submesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        num_links=num_links,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
        allowed_pcc=0.9999,
        all_gather_function=ttnn.experimental.all_gather_async,
        sub_core_grids=sub_core_grids,
    )
    ttnn.ReadDeviceProfiler(submesh_device)

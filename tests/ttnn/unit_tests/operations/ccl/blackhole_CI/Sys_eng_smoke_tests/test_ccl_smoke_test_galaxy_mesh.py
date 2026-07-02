# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.nightly.t3000.ccl.test_all_gather import run_all_gather_impl
from models.common.utility_functions import skip_for_wormhole_b0, run_for_n_dev
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test


# Test uses 3.932GB of space per device to nearly fill the dram
@run_for_n_dev(32)
@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, cluster_axis",
    [
        (4, [1, 1, 24000, 32768], 3, ttnn.TILE_LAYOUT, 0),
        (8, [1, 1, 20000, 32768], 3, ttnn.TILE_LAYOUT, 1),
    ],
    ids=[
        "row_test",
        "column_test",
    ],
)
@pytest.mark.parametrize(
    "ag_input_dtype",
    [
        ttnn.uint32,
    ],
    ids=[
        "uint_32",
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
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}),
    ],
    indirect=["device_params"],
    ids=["fabric"],
)
def test_ccl_ddr_smoke_test(
    bh_2d_mesh_device,
    num_devices,
    ag_output_shape,
    cluster_axis,
    dim,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    num_iters,
):
    validate_test(num_devices, None, bh_2d_mesh_device.shape, cluster_axis)
    # Check all the rows and columns independently within the device
    if cluster_axis == 0:
        submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    else:
        submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))
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


@run_for_n_dev(32)
@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, cluster_axis, ag_input_dtype",
    [
        (4, [1, 1, 6016, 8192], 3, ttnn.TILE_LAYOUT, 0, ttnn.bfloat16),
        (8, [1, 1, 6016, 4096], 3, ttnn.TILE_LAYOUT, 1, ttnn.uint32),
        (8, [1, 1, 6016, 4096], 3, ttnn.TILE_LAYOUT, 1, ttnn.bfloat8_b),
    ],
    ids=[
        "horizontal_test_bfloat16",
        "vertical_test_uint32",
        "vertical_test_bfloat8",
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
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}),
    ],
    indirect=["device_params"],
    ids=["fabric"],
)
def test_ccl_other_smoke_test(
    bh_2d_mesh_device,
    num_devices,
    ag_output_shape,
    cluster_axis,
    dim,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    num_iters,
):
    validate_test(num_devices, None, bh_2d_mesh_device.shape, cluster_axis)
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
            num_l1_banks=110,
        )
    ttnn.ReadDeviceProfiler(submesh_device)

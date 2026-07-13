# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.nightly.t3000.ccl.test_all_gather import run_all_gather_impl
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test
from models.common.utility_functions import skip_for_blackhole, skip_for_wormhole_b0


@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout",
    [
        (4, [1, 1, 10000, 32768], 3, ttnn.TILE_LAYOUT),
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
        (False, 2),
    ],
    ids=["non-trace"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}),
    ],
    indirect=["device_params"],
    ids=["ring"],
)
@pytest.mark.parametrize(
    "cluster_axis",
    [
        0,
    ],
    ids=["row"],
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
    if ttnn.get_num_devices() != 4 and ttnn.get_fabric_config() == ttnn.FabricConfig.FABRIC_1D_RING:
        pytest.skip("This test is only for the quietbox")
    validate_test(num_devices, None, bh_2d_mesh_device.shape, cluster_axis)
    # Check all the rows and columns independantly within the device
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


@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout",
    [
        (4, [1, 1, 6016, 4096], 3, ttnn.TILE_LAYOUT),
        (2, [1, 1, 6016, 2048], 3, ttnn.TILE_LAYOUT),
    ],
    ids=["4 device line", "2_device_line"],
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
        (True, 1),
        (False, 1),
    ],
    ids=["trace", "non-trace"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}),
    ],
    indirect=["device_params"],
    ids=["line"],
)
@pytest.mark.parametrize(
    "cluster_axis",
    [
        0,
        1,
    ],
    ids=["row", "column"],
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
    if ttnn.get_num_devices() != 4 and ttnn.get_fabric_config() == ttnn.FabricConfig.FABRIC_1D_RING:
        pytest.skip("Skipping unsupported case Ring on 2D mesh with no wraparound rings")
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
            num_l1_banks=120,
        )
    ttnn.ReadDeviceProfiler(submesh_device)

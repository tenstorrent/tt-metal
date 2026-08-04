# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.nightly.t3000.ccl.test_all_gather import run_all_gather_impl
from models.common.utility_functions import skip_for_wormhole_b0, run_for_n_dev
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test


# Test uses 3.932GB of space per device to nearly fill the dram
@run_for_n_dev(4)
@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout",
    [
        (4, [1, 1, 24000, 32768], 3, ttnn.TILE_LAYOUT),
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
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}),
    ],
    indirect=["device_params"],
    ids=["fabric"],
)
def test_ccl_ddr_smoke_test(
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
    # Check all the rows and columns independantly within the device
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
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
        cluster_axis=0,
        allowed_pcc=0.9999,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


# P300 with 2 harvested columns so 110 cores are available.
# Test utilizes 1'478'492.16 bytes per core to nearly maximize 1.5MB size
@run_for_n_dev(4)
@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout",
    [
        (4, [1, 1, 6016, 8192], 3, ttnn.TILE_LAYOUT),
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
def test_ccl_other_smoke_test(
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
        cluster_axis=0,
        allowed_pcc=0.9999,
        num_l1_banks=120,
    )
    ttnn.ReadDeviceProfiler(submesh_device)

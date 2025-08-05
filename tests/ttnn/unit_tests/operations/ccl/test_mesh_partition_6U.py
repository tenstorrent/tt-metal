# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import ttnn

# Import the test function from the t3000 file
from tests.ttnn.unit_tests.operations.ccl.test_mesh_partition_t3000 import (
    run_mesh_partition_test,
)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 18432,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [True, False])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((8, 4), (8, 4), id="8x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("per_device_output_shape, dim", [((1, 1, 8, 7168), 2)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("cluster_axis", [0, 1, None])
@pytest.mark.parametrize("mesh_axes", [[0, 1]])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_mesh_partition_rm(
    mesh_device,
    mesh_shape,
    trace_mode,
    per_device_output_shape,
    dtype,
    layout,
    dim,
    cluster_axis,
    mesh_axes,
    input_memory_config,
    output_memory_config,
):
    num_iters = 2
    warmup_iters = 0

    run_mesh_partition_test(
        mesh_device,
        per_device_output_shape,
        dim,
        num_iters,
        warmup_iters,
        trace_mode,
        dtype,
        layout,
        cluster_axis,
        mesh_axes,
        mesh_shape,
        input_memory_config,
        output_memory_config,
        scheme="random",
    )

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.ccl.test_reduce_scatter_TG_nightly import (
    run_line_reduce_scatter_on_TG_with_mesh_tensor_along_rows,
)


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_output_shape, dim, layout",
    [
        (4, 2, [1, 4, 32, 2304], 1, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "buffer_type",
    [
        ttnn.BufferType.DRAM,
        ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize(
    "use_persistent_output",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize("replication_factor", [8])  # 1, 8])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 10281600, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_line_reduce_scatter_on_TG_rows_post_commit(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    buffer_type,
    function_level_defaults,
    replication_factor,
    use_persistent_output,
    num_iters=16,
):
    if mesh_device.get_num_devices() != 32:
        pytest.skip("Not TG!")
    run_line_reduce_scatter_on_TG_with_mesh_tensor_along_rows(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        ttnn.TensorMemoryLayout.INTERLEAVED,
        dim,
        num_links,
        math_op,
        input_dtype,
        layout,
        buffer_type,
        function_level_defaults,
        num_iters=num_iters,
        num_reduce_scatter_instances=replication_factor,
        cluster_axis=1,
        use_reduce_scatter_async=True,
        use_persistent_output=use_persistent_output,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_output_shape, dim, layout",
    [
        (8, 2, [1, 8, 32, 1280], 1, ttnn.TILE_LAYOUT),
        (8, 2, [8, 1, 32, 1280], 0, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "buffer_type",
    [
        ttnn.BufferType.DRAM,
    ],
)
@pytest.mark.parametrize(
    "use_persistent_output",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize("replication_factor", [4])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_line_reduce_scatter_on_TG_cols_post_commit(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    buffer_type,
    function_level_defaults,
    replication_factor,
    use_persistent_output,
    num_iters=16,
):
    if mesh_device.get_num_devices() != 32:
        pytest.skip("Not TG!")

    run_line_reduce_scatter_on_TG_with_mesh_tensor_along_rows(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        ttnn.TensorMemoryLayout.INTERLEAVED,
        dim,
        num_links,
        math_op,
        input_dtype,
        layout,
        buffer_type,
        function_level_defaults,
        num_iters=num_iters,
        num_reduce_scatter_instances=replication_factor,
        cluster_axis=0,
        use_reduce_scatter_async=True,
        use_persistent_output=use_persistent_output,
    )

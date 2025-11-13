# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
import math
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.nightly.tg.ccl.test_all_reduce_async import run_all_reduce_with_mesh_tensor_along_row


# Enumerate the post-commit cases explicitly
@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_output_shape, layout",
    [
        (4, 2, [1, 4, 32, 2304], ttnn.TILE_LAYOUT),
        (4, 2, [4, 1, 64, 1024], ttnn.TILE_LAYOUT),
        (4, 2, [3, 2, 90, 2040], ttnn.TILE_LAYOUT),
        (4, 2, [16, 1, 16, 512], ttnn.ROW_MAJOR_LAYOUT),
        (4, 2, [1, 1, 250, 2048], ttnn.ROW_MAJOR_LAYOUT),
        (4, 2, [2, 2, 350, 350], ttnn.ROW_MAJOR_LAYOUT),
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
        ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize("replication_factor", [8])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_line_all_reduce_on_TG_rows_post_commit(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    num_links,
    math_op,
    input_dtype,
    layout,
    buffer_type,
    function_level_defaults,
    replication_factor,
    num_iters=16,
):
    if mesh_device.get_num_devices() != 32:
        pytest.skip("Not TG!")

    run_all_reduce_with_mesh_tensor_along_row(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        num_links,
        math_op,
        input_dtype,
        layout,
        buffer_type,
        function_level_defaults,
        num_iters=num_iters,
        num_all_reduce_instances=replication_factor,
        cluster_axis=1,
    )


@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_output_shape, layout",
    [
        (8, 1, [1, 8, 32, 1280], ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("replication_factor", [4])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_line_all_reduce_on_TG_cols_post_commit(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    num_links,
    math_op,
    input_dtype,
    layout,
    buffer_type,
    function_level_defaults,
    replication_factor,
    num_iters=16,
):
    if mesh_device.get_num_devices() != 32:
        pytest.skip("Not TG!")

    run_all_reduce_with_mesh_tensor_along_row(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        num_links,
        math_op,
        input_dtype,
        layout,
        buffer_type,
        function_level_defaults,
        num_iters=num_iters,
        num_all_reduce_instances=replication_factor,
        cluster_axis=0,
    )


@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_output_shape, layout",
    [
        (8, 3, [1, 1, 4096, 50304], ttnn.TILE_LAYOUT),
        (8, 3, [1, 1, 2048, 50304], ttnn.TILE_LAYOUT),
        (8, 3, [1, 1, 50304, 4096], ttnn.TILE_LAYOUT),
        (8, 3, [1, 1, 50304, 2048], ttnn.TILE_LAYOUT),
        (8, 3, [1, 1, 50304, 1024], ttnn.TILE_LAYOUT),
        (8, 3, [1, 1, 128000, 4096], ttnn.TILE_LAYOUT),
        (8, 3, [1, 1, 4096, 128000], ttnn.TILE_LAYOUT),
        (8, 3, [1, 1, 1024, 50304], ttnn.TILE_LAYOUT),
        (8, 3, [1, 1, 33, 66], ttnn.TILE_LAYOUT),
        (8, 3, [1, 1, 4094, 50300], ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("replication_factor", [4])
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("mesh_device", [pytest.param((4, 8), id="4x8_grid")], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_line_all_reduce_training(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    num_links,
    math_op,
    input_dtype,
    layout,
    buffer_type,
    function_level_defaults,
    replication_factor,
    num_iters=1,
):
    if mesh_device.get_num_devices() != 32:
        pytest.skip("Not TG!")

    run_all_reduce_with_mesh_tensor_along_row(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        num_links,
        math_op,
        input_dtype,
        layout,
        buffer_type,
        function_level_defaults,
        num_iters=num_iters,
        num_all_reduce_instances=replication_factor,
        cluster_axis=1,
    )

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull, get_devices_for_t3000
import itertools
from ttnn import ShardTensorToMesh
from tests.ttnn.unit_tests.operations.test_all_gather import run_all_gather_sharded


@pytest.mark.timeout(120)
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_devices", [8])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT])
# @pytest.mark.parametrize("num_cores", [1])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,  # https://github.com/tenstorrent/tt-metal/issues/9686
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "tensor_mem_layout",
    [
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ],
)
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize(
    "input_shape, input_shard_shape,shard_grid",
    (
        # LLama
        (
            (1, 1, 32, 1024),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
    ),
)
@pytest.mark.parametrize(
    "n_buffer",
    (
        # LLama
        1,
        2,
        3,
        4,
        6,
        8,
    ),
)
@pytest.mark.parametrize(
    "n_worker",
    (
        # LLama
        2,
        4,
        6,
        8,
        10,
        12,
    ),
)
@pytest.mark.parametrize("num_iter", [1000])
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 17068032}], indirect=True)
def test_all_gather_sharded_post_commit(
    t3k_mesh_device,
    num_devices,
    input_shape,
    input_shard_shape,
    shard_grid,
    dim,
    num_links,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    n_worker,
    n_buffer,
    # num_cores,
    use_program_cache,
    function_level_defaults,
    enable_async,
    num_iter,
):
    logger.info(f"Running for n_worker={n_worker}, n_buffer={n_buffer}:")
    run_all_gather_sharded(
        t3k_mesh_device,
        num_devices,
        input_shape,
        input_shard_shape,
        shard_grid,
        dim,
        num_links,
        orientation,
        input_dtype,
        tensor_layout,
        tensor_mem_layout,
        # num_cores,
        use_program_cache,
        function_level_defaults,
        all_gather_operation=ttnn.all_gather,
        enable_async=enable_async,
        n_worker=n_worker,
        n_buffer=n_buffer,
        num_iter=num_iter,
        trace_mode=True,
    )

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.test_reduce_scatter_post_commit import run_reduce_scatter_test


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (2, 1),
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape, scatter_dim, layout",
    [
        ([1, 1, 32, 4096], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 2048], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 768, 4096], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 768, 2048], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 768, 1024], 3, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("enable_async", [True])
def test_ring_reduce_scatter_post_commit(
    n300_mesh_device,
    num_devices,
    per_chip_output_shape,
    scatter_dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    enable_async,
    num_iters=5,
):
    run_reduce_scatter_test(
        n300_mesh_device,
        num_devices,
        per_chip_output_shape,
        scatter_dim,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        num_iters=num_iters,
        enable_async=enable_async,
    )

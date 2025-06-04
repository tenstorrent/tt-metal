# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from tests.ttnn.unit_tests.operations.ccl.test_reduce_scatter_post_commit import (
    run_reduce_scatter_test,
)
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (8, 1),
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape, dim, layout",
    [
        ([1, 8, 1024, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 4, 1024, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 4, 2048, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 32], -1, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 64], -1, ttnn.TILE_LAYOUT),
        ([1, 1, 64, 64], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 128], -1, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 256], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 512], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 2048], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 128, 1024], -1, ttnn.TILE_LAYOUT),
        ([1, 1, 128, 8192], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 2048, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 2048, 8192], -1, ttnn.TILE_LAYOUT),
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
def test_reduce_scatter_t3k_8chip_nightly(
    t3k_mesh_device,
    num_devices,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    num_iters=1,
):
    run_reduce_scatter_test(
        t3k_mesh_device,
        num_devices,
        per_chip_output_shape,
        dim,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        num_iters=num_iters,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (4, 2),
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape, dim, layout",
    [
        ([1, 8, 1024, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 4, 1024, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 4, 2048, 1024], -1, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 512], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 2048], -1, ttnn.TILE_LAYOUT),
        ([1, 1, 128, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 128, 8192], -1, ttnn.TILE_LAYOUT),
        ([1, 1, 2048, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 2048, 8192], 3, ttnn.TILE_LAYOUT),
        # These shapes result in some workers with no work, which is currently
        # not functional due to hang. Issue tracked at: https://github.com/tenstorrent/tt-metal/issues/14004
        # ([1, 1, 32, 32], 3, ttnn.TILE_LAYOUT),
        # ([1, 1, 32, 64], 3, ttnn.TILE_LAYOUT),
        # ([1, 1, 64, 64], 3, ttnn.TILE_LAYOUT),
        # ([1, 1, 32, 128], 3, ttnn.TILE_LAYOUT),
        # ([1, 1, 32, 256], 3, ttnn.TILE_LAYOUT),
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
def test_reduce_scatter_t3k_4chip_nightly(
    pcie_mesh_device,
    num_devices,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    num_iters=1,
):
    run_reduce_scatter_test(
        pcie_mesh_device,
        num_devices,
        per_chip_output_shape,
        dim,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        num_iters=num_iters,
    )

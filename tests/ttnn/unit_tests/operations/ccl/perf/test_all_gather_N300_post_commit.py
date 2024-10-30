# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.ccl.test_all_gather import (
    run_all_gather_on_n300_impl,
)


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, output_shape, dim, layout",
    [
        (2, 1, [1, 1, 64, 16384], 3, ttnn.TILE_LAYOUT),
        (2, 1, [8, 5, 32, 768], 3, ttnn.TILE_LAYOUT),
        (2, 1, [1, 1, 32, 736], 3, ttnn.TILE_LAYOUT),
        (2, 1, [1, 1, 32, 704], 3, ttnn.TILE_LAYOUT),
        (2, 1, [1, 1, 64, 704], 3, ttnn.TILE_LAYOUT),
        (2, 1, [1, 1, 32, 736], 3, ttnn.ROW_MAJOR_LAYOUT),
        (2, 1, [1, 1, 32, 704], 3, ttnn.ROW_MAJOR_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
    ],
)
@pytest.mark.parametrize("num_iters", [1])
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 266240}], indirect=True)
def test_all_gather_on_n300_post_commit(
    n300_mesh_device,
    num_devices,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_all_gather_on_n300_impl(
        n300_mesh_device,
        num_devices,
        output_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Ring,
        num_iters=num_iters,
        enable_async=enable_async,
        trace_mode=True,
    )

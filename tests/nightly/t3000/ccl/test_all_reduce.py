# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
import math
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.unit_tests.operations.ccl.test_all_reduce_async import run_all_reduce_test


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (4, 1),
        # (8, 1), # skipped as 8 devices result in hang in all gather
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape",
    [
        ([1, 1, 32, 4096]),
        ([1, 1, 32, 8192]),
        ([1, 1, 32, 1024]),
        ([1, 1, 32, 2048]),
        ([1, 1, 4096, 32]),
        # ([1, 1, 8192, 32]), # skipped as it hangs in reduce scatter part.
        ([1, 1, 1024, 32]),
        ([1, 1, 2048, 32]),
        ([4, 1, 32, 4096]),
        ([8, 1, 32, 1024]),
        ([1, 4, 1024, 32]),
        ([2, 4, 2048, 32]),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
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
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_ring_all_reduce_post_commit(
    t3k_mesh_device,
    num_devices,
    per_chip_output_shape,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    function_level_defaults,
    num_iters=2,
):
    run_all_reduce_test(
        t3k_mesh_device,
        num_devices,
        per_chip_output_shape,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        function_level_defaults,
        num_iters=num_iters,
    )


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (2, 1),
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape",
    [
        ([2, 2, 64, 64]),
        ([1, 1, 64, 64]),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
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
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_ring_all_reduce_post_commit_2chip(
    t3k_mesh_device,
    num_devices,
    per_chip_output_shape,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    function_level_defaults,
    num_iters=2,
):
    run_all_reduce_test(
        t3k_mesh_device,
        num_devices,
        per_chip_output_shape,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        function_level_defaults,
        num_iters=num_iters,
        topology=ttnn.Topology.Linear,
    )

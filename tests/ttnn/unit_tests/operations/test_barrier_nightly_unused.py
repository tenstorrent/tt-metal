# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.unit_tests.operations.test_barrier_post_commit import (
    is_unsupported_case,
    run_barrier_test,
)
from models.utility_functions import skip_for_grayskull, get_devices_for_t3000
import itertools


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (8, 1),
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape, scatter_dim, layout",
    [
        ([1, 4, 2048, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 32], 3, ttnn.TILE_LAYOUT),
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
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("enable_async", [True, False])
def test_barrier_nightly(
    t3k_mesh_device,
    num_devices,
    per_chip_output_shape,
    scatter_dim,
    input_dtype,
    layout,
    mem_config,
    enable_async,
    num_iters=1,
):
    run_barrier_test(
        t3k_mesh_device,
        num_devices,
        per_chip_output_shape,
        scatter_dim,
        input_dtype,
        layout,
        mem_config,
        num_iters=num_iters,
        enable_async=enable_async,
    )


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (4, 2),
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape, scatter_dim, layout",
    [
        ([1, 4, 2048, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 32], 3, ttnn.TILE_LAYOUT),
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
    ],
)
@pytest.mark.parametrize("enable_async", [True, False])
def test_barrier_nightly(
    pcie_mesh_device,
    num_devices,
    per_chip_output_shape,
    scatter_dim,
    input_dtype,
    layout,
    mem_config,
    num_links,
    enable_async,
    num_iters=1,
):
    run_barrier_test(
        pcie_mesh_device,
        num_devices,
        per_chip_output_shape,
        scatter_dim,
        input_dtype,
        layout,
        mem_config,
        num_iters=num_iters,
        enable_async=enable_async,
    )

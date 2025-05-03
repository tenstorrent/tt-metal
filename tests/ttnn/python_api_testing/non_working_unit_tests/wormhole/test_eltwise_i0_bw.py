# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.sweep_framework.sweeps.eltwise.unary_backward.i0_bw.i0_bw import run as run_test


test_sweep_args = [
    (
        (4, 7, 21, 133),
        [ttnn.bfloat16, ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        16305027,
    ),
    (
        (4, 7, 21, 133),
        [ttnn.bfloat16, ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        3624344,
    ),
    (
        (4, 7, 21, 133),
        [ttnn.bfloat16, ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        3624344,
    ),
    (
        (4, 7, 21, 133),
        [ttnn.bfloat16, ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.L1_MEMORY_CONFIG),
        3624344,
    ),
    (
        (4, 7, 21, 133),
        [ttnn.bfloat16, ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        3624344,
    ),
    (
        (4, 6, 105, 245),
        [ttnn.bfloat16, ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        16305027,
    ),
    (
        (4, 6, 105, 245),
        [ttnn.bfloat16, ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        3624344,
    ),
    (
        (4, 6, 105, 245),
        [ttnn.bfloat16, ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        3624344,
    ),
    (
        (4, 6, 105, 245),
        [ttnn.bfloat16, ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.L1_MEMORY_CONFIG),
        3624344,
    ),
    (
        (4, 6, 105, 245),
        [ttnn.bfloat16, ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        3624344,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_eltwise_i0_bw(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    random.seed(0)

    [pcc, e2e_perf] = run_test(
        input_shape, dtype[0], dtype[1], dlayout[0], in_mem_config[0], in_mem_config[1], out_mem_config, device=device
    )
    [passed, pcc_value] = pcc

    assert passed, f"pcc={pcc[1]}"

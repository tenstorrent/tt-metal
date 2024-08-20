# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import permute as tt_permute
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand


def run_permute_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, permute_dims, device):
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = gen_rand(size=input_shape, low=-100, high=100).to(torch.bfloat16)
    # compute ref value
    x_ref = x.detach().clone()
    ref_value = pytorch_ops.permute(x_ref, permute_dims=permute_dims)

    tt_result = tt_permute(
        x=x,
        permute_dims=permute_dims,
        device=device,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_equal(ref_value, tt_result)
    logger.debug(pcc_value)
    logger.debug(success)

    assert success


test_sweep_args = [
    (
        (2, 4, 32, 64),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        "SYSTEM_MEMORY",
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        16305027,
        (3, 1, 2, 0),
    ),
    (
        (3, 6, 32, 64),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        11271489,
        (0, 2, 3, 1),
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, permute_dims",
    (test_sweep_args),
)
def test_permute_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, permute_dims, device):
    random.seed(0)
    run_permute_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, permute_dims, device)

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
from tests.tt_eager.python_api_testing.sweep_tests import tt_lib_ops
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand_inf


def run_eltwise_isneginf_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    x = gen_rand_inf(size=input_shape, low=-100, high=100)

    # compute ref value
    ref_value = pytorch_ops.isneginf(x=x)

    tt_result = tt_lib_ops.eltwise_isneginf(
        x=x,
        device=device,
        dtype=dtype,
        layout=dlayout,
        input_mem_config=in_mem_config,
        output_mem_config=out_mem_config,
    )

    success, pcc_value = comp_equal(ref_value, tt_result)
    logger.debug(pcc_value)
    logger.debug(success)

    assert success


test_sweep_args = [
    (
        (5, 10, 35, 162),
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)],
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        3624344,
    ),
    (
        (6, 12, 39, 22),
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)],
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        7965166,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_eltwise_isneginf(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    random.seed(0)
    run_eltwise_isneginf_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)

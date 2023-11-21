# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_logaddexp as tt_logaddexp
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops


def run_matmul_test(input_shape_1, input_shape_2, dtype, dlayout, in_mem_config, out_mem_config, low_b, high_b, device):
    torch.manual_seed(0)

    if in_mem_config[0] == "SYSTEM_MEMORY":
        in_mem_config[0] = None

    if in_mem_config[1] == "SYSTEM_MEMORY":
        in_mem_config[1] = None

    x = gen_rand(size=input_shape_1, low=low_b, high=high_b)
    y = gen_rand(size=input_shape_2, low=low_b, high=high_b)

    tt_result = tt_logaddexp(
        x=x,
        y=y,
        device=device,
        dtype=dtype,
        layout=dlayout,
        input_mem_config=in_mem_config,
        output_mem_config=out_mem_config,
    )

    ref_value = pytorch_ops.silu(x)

    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)
    logger.debug(success)


test_sweep_args = [
    (
        (1, 4, 128, 128),
        (1, 4, 128, 128),
        [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
        [
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
    ),
]


@pytest.mark.parametrize(
    "input_shape_1, input_shape_2, dtype, dlayout, in_mem_config, out_mem_config",
    (test_sweep_args),
)
def test_matmul_test(input_shape_1, input_shape_2, dtype, dlayout, in_mem_config, out_mem_config, device):
    random.seed(0)
    a = input_shape_1[3]
    high_b = 10
    low_b = 0

    for i in range(0, 10):
        input_shape_1x = [1, 4, 128, 128]
        input_shape_2x = [1, 4, 128, 128]

        logger.info(low_b)
        logger.info(high_b)

        run_matmul_test(
            input_shape_1x, input_shape_2x, dtype, dlayout, in_mem_config, out_mem_config, low_b, high_b, device
        )
        high_b = high_b + 10
        low_b = low_b + 10

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_logaddexp as tt_eltwise_logaddexp
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand
from tests.tt_eager.python_api_testing.sweep_tests.common import set_slow_dispatch_mode


def run_eltwise_logaddexp_test(
    input_shape_1, input_shape_2, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode, device
):
    torch.manual_seed(data_seed)
    prev_dispatch_mode = set_slow_dispatch_mode(dispatch_mode)

    x = gen_rand(size=input_shape_1, low=-64, high=64)
    y = gen_rand(size=input_shape_2, low=-64, high=64)
    # compute ref value
    x_ref = x.detach().clone()
    y_ref = y.detach().clone()

    ref_value = pytorch_ops.logaddexp(x_ref, y_ref)

    if in_mem_config[0] == "SYSTEM_MEMORY":
        in_mem_config[0] = None

    if in_mem_config[1] == "SYSTEM_MEMORY":
        in_mem_config[1] = None

    tt_result = tt_eltwise_logaddexp(
        x=x,
        y=y,
        device=device,
        dtype=dtype,
        layout=dlayout,
        input_mem_config=in_mem_config,
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)
    logger.debug(success)

    set_slow_dispatch_mode(prev_dispatch_mode)
    assert success


test_sweep_args = [
    (
        (2, 5, 64, 224),
        (2, 5, 64, 224),
        [ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT8_B],
        [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
        [
            "SYSTEM_MEMORY",
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        2474385,
        "",
    ),
    (
        (4, 7, 32, 96),
        (4, 7, 32, 96),
        [ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
        [
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        17155532,
        "",
    ),
    (
        (2, 11, 160, 224),
        (2, 11, 160, 224),
        [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B],
        [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
        [
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        14073508,
        "1",
    ),
]


@pytest.mark.parametrize(
    "input_shape_1, input_shape_2, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode",
    (test_sweep_args),
)
def test_eltwise_logaddexp_test(
    input_shape_1, input_shape_2, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode, device
):
    random.seed(0)
    run_eltwise_logaddexp_test(
        input_shape_1, input_shape_2, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode, device
    )

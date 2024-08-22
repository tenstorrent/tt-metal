# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import random
from loguru import logger
import pytest
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops, tt_lib_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand


def run_reshape_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, reshape_dims, device):
    random.seed(0)
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = gen_rand(size=input_shape, low=-100, high=100)
    x_ref = x.detach().clone()

    # compute ref value
    ref_value = pytorch_ops.reshape(x=x_ref, reshape_dims=reshape_dims)

    tt_result = tt_lib_ops.reshape(
        x=x,
        reshape_dims=reshape_dims,
        device=device,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)

    assert success


test_sweep_args = [
    (
        (2, 3, 256, 200),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        None,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        16241115,
        [30, 1, 320, 32],
    ),
    (
        (3, 6, 48, 192),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        None,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        16241115,
        [27, 1, 192, 32],
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, reshape_dims",
    (test_sweep_args),
)
def test_reshape(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, reshape_dims, device):
    run_reshape_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, reshape_dims, device)

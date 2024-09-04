# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import reduce_sum_h as tt_reduce_sum_h


def run_reduce_max_h_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = torch.Tensor(size=input_shape).uniform_(-100, 100).to(torch.bfloat16)
    x_ref = x.detach().clone()

    # get ref result
    ref_value = pytorch_ops.reduce_sum(x_ref, dims=(-2,))

    tt_result = tt_reduce_sum_h(
        x=x,
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
        (1, 4, 20, 166),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        "SYSTEM_MEMORY",
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        14632035,
    ),
    (
        (4, 1, 154, 26),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        "SYSTEM_MEMORY",
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        8728623,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_reduce_max_h_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_reduce_max_h_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)

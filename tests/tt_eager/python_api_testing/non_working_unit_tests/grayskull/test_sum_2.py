# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import random
from loguru import logger
import pytest
import torch
import tt_lib as ttl


from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests import tt_lib_ops, pytorch_ops
from models.utility_functions import tt2torch_tensor
from tests.tt_eager.python_api_testing.sweep_tests.common import set_slow_dispatch_mode


def run_sum_2_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    random.seed(0)
    torch.manual_seed(data_seed)

    if in_mem_config[0] == "SYSTEM_MEMORY":
        in_mem_config[0] = None

    x = torch.Tensor(size=input_shape).uniform_(-100, 100)

    ref_value = pytorch_ops.sum(x, dim=2)

    tt_result = tt_lib_ops.sum(
        x=x,
        dim=2,
        device=device,
        dtype=dtype,
        layout=dlayout,
        input_mem_config=in_mem_config,
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)

    assert success


test_sweep_args = [
    (
        (6, 2, 216, 186),
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.ROW_MAJOR],
        ["SYSTEM_MEMORY"],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        13482735,
    ),
    (
        (6, 1, 140, 110),
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.ROW_MAJOR],
        ["SYSTEM_MEMORY"],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        2348954,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed",
    (test_sweep_args),
)
def test_sum_2_test(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, device):
    run_sum_2_tests(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, device)

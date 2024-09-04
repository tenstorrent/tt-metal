# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.pytorch_ops import relu_min as pt_relu_min
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_relu_min as tt_relu_min


def run_eltwise_relu_min_tests(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, lower_limit, data_seed, device
):
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape).uniform_(-100, 100).to(torch.bfloat16)
    x_ref = x.detach().clone()

    # get referent value
    ref_value = pt_relu_min(x_ref, lower_limit=lower_limit)

    # calculate tt output
    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    tt_result = tt_relu_min(
        x=x,
        lower_limit=lower_limit,
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
        (4, 2, 96, 160),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        "SYSTEM_MEMORY",
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        99.5,
        12915139,
    ),
    (
        (3, 10, 64, 96),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        97.5,
        8726038,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, lower_limit, data_seed",
    (test_sweep_args),
)
def test_eltwise_relu_min_test(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, lower_limit, data_seed, device
):
    run_eltwise_relu_min_tests(
        input_shape, dtype, dlayout, in_mem_config, out_mem_config, lower_limit, data_seed, device
    )

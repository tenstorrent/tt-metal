# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_logical_and_ as tt_eltwise_logical_and_


def run_eltwise_logical_andi_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    out_mem_config,
    data_seed,
    device,
):
    random.seed(0)
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = torch.Tensor(size=input_shape).uniform_(-100, 100).to(torch.bfloat16)
    y = torch.Tensor(size=input_shape).uniform_(-100, 100).to(torch.bfloat16)
    x_ref = x.detach().clone()
    y_ref = y.detach().clone()

    # get referent value
    golden_function = ttnn.get_golden_function(ttnn.logical_and_)
    ref_value = golden_function(x_ref, y_ref)

    # calculate tt output
    logger.info("Running eltwise_and_ test")
    tt_result = tt_eltwise_logical_and_(
        x=x,
        y=y,
        device=device,
        dtype=dtype,
        layout=dlayout,
        input_mem_config=in_mem_config,
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, x)
    logger.debug(pcc_value)

    assert success


test_sweep_args = [
    (
        (6, 9, 192, 128),
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT],
        [
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ],
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        14854324,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_eltwise_logical_andi_test(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    out_mem_config,
    data_seed,
    device,
):
    run_eltwise_logical_andi_tests(
        input_shape,
        dtype,
        dlayout,
        in_mem_config,
        out_mem_config,
        data_seed,
        device,
    )

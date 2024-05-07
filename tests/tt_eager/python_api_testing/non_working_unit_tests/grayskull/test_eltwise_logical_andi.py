# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.pytorch_ops import ne as pt_ne
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_logical_andi as tt_eltwise_logical_andi


def run_eltwise_logical_andi_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    out_mem_config,
    immediate,
    data_seed,
    device,
):
    random.seed(0)
    torch.manual_seed(data_seed)

    if in_mem_config[0] == "SYSTEM_MEMORY":
        input0_mem_config[0] = None

    if in_mem_config[1] == "SYSTEM_MEMORY":
        input0_mem_config[1] = None

    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)
    x_ref = x.detach().clone()

    # get referent value
    ref_value = torch.logical_and(x_ref, immediate)

    # calculate tt output
    logger.info("Running eltwise_andi test")
    tt_result = tt_eltwise_logical_andi(
        x=x,
        immediate=immediate,
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
        [(3, 9, 96, 192), (3, 9, 96, 192)],
        [ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT8_B],
        [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
        [
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            "SYSTEM_MEMORY",
        ],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        0,
        10638326,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, immediate, data_seed",
    (test_sweep_args),
)
def test_eltwise_logical_andi_test(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    out_mem_config,
    immediate,
    data_seed,
    device,
):
    run_eltwise_logical_andi_tests(
        input_shape,
        dtype,
        dlayout,
        in_mem_config,
        out_mem_config,
        immediate,
        data_seed,
        device,
    )

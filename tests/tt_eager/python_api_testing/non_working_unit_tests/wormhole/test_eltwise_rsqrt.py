# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_rsqrt as tt_eltwise_rsqrt


def run_eltwise_rsqrt_tests(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, fast_and_approx, data_seed, device
):
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = torch.Tensor(size=input_shape).uniform_(-100, 100)
    x_ref = x.detach().clone()

    # get ref result
    ref_value = pytorch_ops.rsqrt(x_ref)

    tt_result = tt_eltwise_rsqrt(
        x=x,
        fast_and_approx=fast_and_approx,
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
        (4, 7, 32, 96),
        ttl.tensor.DataType.BFLOAT8_B,
        ttl.tensor.Layout.TILE,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        True,
        19575052,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, fast_and_approx, data_seed",
    (test_sweep_args),
)
def test_eltwise_rsqrt_test(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, fast_and_approx, data_seed, device
):
    run_eltwise_rsqrt_tests(
        input_shape, dtype, dlayout, in_mem_config, out_mem_config, fast_and_approx, data_seed, device
    )

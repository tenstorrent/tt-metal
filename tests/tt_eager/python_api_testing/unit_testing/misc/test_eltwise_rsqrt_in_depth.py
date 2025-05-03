# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_rsqrt as tt_eltwise_rsqrt


def run_eltwise_rsqrt_tests(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, range_start, range_end, data_seed, device
):
    torch.manual_seed(data_seed)

    fast_and_appx = False

    x = torch.Tensor(size=input_shape).uniform_(range_start, range_end)
    x_ref = x.detach().clone()

    # compute ref value
    ref_value = pytorch_ops.rsqrt(x_ref)

    tt_result = tt_eltwise_rsqrt(
        x=x,
        fast_and_approx=fast_and_appx,
        device=device,
        device_id=0,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)

    assert success, f"PCC check failed for range {range_start}:{range_end}. PCC: {pcc_value}"


test_sweep_args = []
x = 1.4
inc = 0.5

while x < 10:
    y = round(x + inc, 1)

    test_sweep_args.append(
        (
            (3, 11, 92, 100),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            x,
            y,
            5147678,
        )
    )

    x = y

    if x >= 5:
        inc = 1


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, range_start, range_end, data_seed",
    (test_sweep_args),
)
def test_eltwise_rsqrt_test(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, range_start, range_end, data_seed, device
):
    run_eltwise_rsqrt_tests(
        input_shape, dtype, dlayout, in_mem_config, out_mem_config, range_start, range_end, data_seed, device
    )

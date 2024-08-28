# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_argmax_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    **kwargs,
):
    dim = kwargs.get("dim", None)

    torch.manual_seed(data_seed)
    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)

    device = ttnn.open_device(0)

    try:
        # get ref result
        ref_value = torch.argmax(x, dim=dim)

        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])

        tt_result = ttnn.argmax(x, dim=dim, memory_config=output_mem_config)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))

        if dim:
            tt_result = tt_result.squeeze(dim=dim)
        else:
            tt_result = tt_result.squeeze()

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    ttnn.close_device(device)

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        [(1, 3, 19, 98)],
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        3313947,
        {"dim": 3},
    ),
    (
        [(2, 6, 107, 102)],
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        [ttnn.L1_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        19042500,
        {"dim": 3},
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, kwargs",
    (test_sweep_args),
)
def test_argmax(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, kwargs):
    run_argmax_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, **kwargs)

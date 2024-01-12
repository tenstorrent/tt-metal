# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, set_slow_dispatch_mode
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_softmax_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    dispatch_mode,
    device,
):
    torch.manual_seed(data_seed)
    prev_dispatch_mode = set_slow_dispatch_mode(dispatch_mode)

    x = torch.Tensor(size=input_shape).uniform_(-1, 1).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = torch.softmax(x, -1)
        x = ttnn_ops.torch_to_ttnn(x, device, dlayout[0], in_mem_config, dtype[0])

        tt_result = ttnn.softmax(x, dim=-1)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        set_slow_dispatch_mode(prev_dispatch_mode)
        raise e

    set_slow_dispatch_mode(prev_dispatch_mode)
    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        (3, 1, 64, 32),
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        79106,
        "1",
    ),
    (
        (3, 7, 64, 32),
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        16899236,
        "1",
    ),
    (
        (3, 10, 192, 64),
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        18539618,
        "1",
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode",
    (test_sweep_args),
)
def test_softmax(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode, device):
    run_softmax_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode, device)

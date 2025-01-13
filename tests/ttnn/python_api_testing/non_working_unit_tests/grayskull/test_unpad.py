# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def ref_unpad(x, *args, output_tensor_start, output_tensor_end, **kwargs):
    out = x[
        output_tensor_start[0] : output_tensor_end[0],
        output_tensor_start[1] : output_tensor_end[1],
        output_tensor_start[2] : output_tensor_end[2],
        output_tensor_start[3] : output_tensor_end[3],
    ]

    return out


def run_unpad_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    output_tensor_start,
    output_tensor_end,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape).uniform_(-100, 100).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = ref_unpad(x, output_tensor_start=output_tensor_start, output_tensor_end=output_tensor_end)
        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config, dtype[0])

        tt_result = ttnn.slice(x, output_tensor_start, output_tensor_end)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)

    except Exception as e:
        logger.warning(f"Operation execution crashed")
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        (5, 1, 133, 170),
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        (0, 0, 0, 0),
        (2, 0, 90, 5),
        16961029,
    ),
    (
        (6, 6, 51, 192),
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        (0, 0, 0, 0),
        (3, 3, 11, 119),
        13612806,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, output_tensor_start, output_tensor_end, data_seed",
    (test_sweep_args),
)
def test_unpad(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    out_mem_config,
    output_tensor_start,
    output_tensor_end,
    data_seed,
    device,
):
    run_unpad_tests(
        input_shape,
        dtype,
        dlayout,
        in_mem_config,
        out_mem_config,
        output_tensor_start,
        output_tensor_end,
        data_seed,
        device,
    )

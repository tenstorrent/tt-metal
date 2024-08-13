# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import random
from loguru import logger
import pytest
import torch
import ttnn.deprecated as ttl

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import arange as tt_arange


def tensor_to_device(x, device, buffer_type):
    if buffer_type == None:
        return x

    return x.to(device, buffer_type)


def run_arange_tests(input_shape, dtype, dlayout, buffer_type, output_mem_config, data_seed, start, end, step, device):
    random.seed(0)
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape).uniform_(-10, 10)
    x_ref = x.detach().clone()
    ref_value = torch.arange(start, end, step)

    ttz = tt_arange(
        x=x,
        start=start,
        end=end,
        step=step,
        device=device,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[
            ttnn.experimental.tensor.MemoryConfig(ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, buffer_type)
        ],
        output_mem_config=output_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ttz, ref_value)
    logger.debug(pcc_value)

    assert success


test_sweep_args = [
    (
        (7, 14, 32, 160),
        ttnn.experimental.tensor.DataType.BFLOAT8_B,
        ttnn.experimental.tensor.Layout.TILE,
        ttnn.experimental.tensor.BufferType.L1,
        ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.DRAM
        ),
        15991940,
        -75,
        -56,
        7,
    ),
    (
        (2, 20, 416, 160),
        ttnn.experimental.tensor.DataType.BFLOAT16,
        ttnn.experimental.tensor.Layout.TILE,
        ttnn.experimental.tensor.BufferType.L1,
        ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.L1
        ),
        18784230,
        41,
        46,
        5,
    ),
    (
        (10, 21, 480, 128),
        ttnn.experimental.tensor.DataType.BFLOAT16,
        ttnn.experimental.tensor.Layout.TILE,
        ttnn.experimental.tensor.BufferType.L1,
        ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.DRAM
        ),
        16005792,
        30,
        94,
        5,
    ),
    (
        (10, 21, 480, 128),
        ttnn.experimental.tensor.DataType.BFLOAT8_B,
        ttnn.experimental.tensor.Layout.TILE,
        ttnn.experimental.tensor.BufferType.DRAM,
        ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.DRAM
        ),
        17493725,
        34,
        71,
        6,
    ),
    (
        (10, 21, 480, 128),
        ttnn.experimental.tensor.DataType.BFLOAT8_B,
        ttnn.experimental.tensor.Layout.TILE,
        ttnn.experimental.tensor.BufferType.L1,
        ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.DRAM
        ),
        8740671,
        38,
        51,
        2,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, buffer_type, out_mem_config, data_seed, start, end, step",
    (test_sweep_args),
)
def test_arange_test(input_shape, dtype, dlayout, buffer_type, out_mem_config, data_seed, start, end, step, device):
    run_arange_tests(input_shape, dtype, dlayout, buffer_type, out_mem_config, data_seed, start, end, step, device)

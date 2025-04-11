# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from loguru import logger
from models.utility_functions import comp_allclose_and_pcc, skip_for_blackhole


def run_fold_test(device, input_shape, output_size, kernel_size, dilation, padding, stride, dtype):
    if dtype == torch.float:
        torch_input = torch.randn(input_shape, dtype=dtype)
    elif dtype == torch.bfloat16:
        # Make bfloat16 input mostly positive by adding 1, thus making output result
        # more positive and futher from 0 to avoid rounding precision problem with bfloat16
        torch_input = torch.randn(input_shape, dtype=dtype) + 1
    torch_fold = torch.nn.Fold(
        output_size=output_size, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride
    )
    expected = torch_fold(torch_input)

    tt_dtype = ttnn.bfloat16 if dtype == torch.bfloat16 else ttnn.float32
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=tt_dtype)
    tt_out = ttnn.operations.moreh.fold(tt_input, None, output_size, kernel_size, dilation, padding, stride)
    actual = ttnn.to_torch(tt_out)
    if dtype == torch.float:
        passing, out = comp_allclose_and_pcc(expected, actual)
    elif dtype == torch.bfloat16:
        passing, out = comp_allclose_and_pcc(expected, actual, rtol=0.05, atol=0.05)
    assert passing


@skip_for_blackhole("Fails on BH. Issue #20577")
@pytest.mark.parametrize(
    "input_shape,output_size,kernel_size,dilation,padding,stride",
    [
        [(32, 32), (7, 11), (4, 4), (1, 1), (0, 0), (1, 1)],  # input single tile 2D
        [(72, 900), (32, 32), (3, 3), (1, 1), (0, 0), (1, 1)],  # input multi tile 2D
        [(128, 324), (32, 32), (4, 4), (2, 2), (5, 5), (2, 2)],  # input multi tile 2D with padding, dilation, stride
        [(1, 32, 32), (7, 11), (4, 4), (1, 1), (0, 0), (1, 1)],  # input single tile
        [(1, 32, 32), (5, 9), (2, 2), (1, 1), (0, 0), (1, 1)],  # input single tile
        [(1, 32, 32), (5, 9), (2, 2), (2, 2), (2, 4), (2, 2)],  # input single tile with padding, dilation, stride
        [(1, 9, 32), (6, 10), (3, 3), (1, 1), (0, 0), (1, 1)],  # small
        [(32, 75, 784), (32, 32), (5, 5), (1, 1), (0, 0), (1, 1)],  # multi tile
        [(32, 27, 100), (12, 12), (3, 3), (1, 1), (0, 0), (1, 1)],  # multi tile
        [(5, 64, 36), (12, 12), (4, 4), (2, 2), (3, 3), (2, 2)],  # multi tile with padding, dilation, stride
        [(5, 144, 42), (14, 16), (6, 6), (2, 2), (4, 4), (2, 2)],  # multi tile with padding, dilation, stride
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float,
        torch.bfloat16,
    ],
)
def test_fold(device, input_shape, output_size, kernel_size, dilation, padding, stride, dtype):
    torch.manual_seed(2024)
    run_fold_test(device, input_shape, output_size, kernel_size, dilation, padding, stride, dtype)


@pytest.mark.parametrize(
    "input_shape,output_size,kernel_size,dilation,padding,stride",
    [
        [(1, 32, 32), (7, 11), (4, 4), (1, 1), (0, 0), (1, 1)],  # input single tile
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float,
        torch.bfloat16,
    ],
)
def test_fold_callback(
    device, input_shape, output_size, kernel_size, dilation, padding, stride, dtype, use_program_cache
):
    torch.manual_seed(0)
    num_program_cache_entries_list = []
    for i in range(2):
        run_fold_test(device, input_shape, output_size, kernel_size, dilation, padding, stride, dtype)
        # Add dummy tensor to make sure that created tensor in 2 iteration don't share the same addr
        torch_dummy = torch.randn([32, 32])
        tt_dummy = ttnn.from_torch(torch_dummy, device=device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]

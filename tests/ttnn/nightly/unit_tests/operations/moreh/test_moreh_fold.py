# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from loguru import logger
from models.common.utility_functions import comp_allclose_and_pcc

# Module-scoped device: opens once per file instead of once per test case.
pytestmark = pytest.mark.use_module_device


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


def test_moreh_fold_golden_accepts_bound_positional_arguments():
    input_tensor = torch.randn(1, 4, 4)
    output_size = (3, 3)
    kernel_size = (2, 2)
    golden_function = ttnn.get_golden_function(ttnn.moreh_fold)

    actual = golden_function(
        input_tensor,
        None,
        output_size,
        kernel_size,
        (1, 1),
        (0, 0),
        (1, 1),
        None,
    )
    expected = torch.nn.functional.fold(input_tensor, output_size=output_size, kernel_size=kernel_size)

    torch.testing.assert_close(actual, expected)


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
def test_fold_callback(device, input_shape, output_size, kernel_size, dilation, padding, stride, dtype):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests in this file.
    device.clear_program_cache()
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


def test_fold_l1_output_short_row(device):
    """Regression test for #57475 item 2: an L1-resident output whose real row is shorter than
    the 32-byte-rounded DFB page size used to spill the rounding padding past the row on every
    page write (past the end of the buffer on a bank's last page)."""
    torch.manual_seed(2024)
    input_shape = (1, 4, 21)
    output_size = (4, 8)
    kernel_size = (2, 2)
    dilation = (1, 1)
    padding = (0, 0)
    stride = (1, 1)

    torch_input = torch.randn(input_shape, dtype=torch.bfloat16) + 1
    torch_fold = torch.nn.Fold(
        output_size=output_size, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride
    )
    expected = torch_fold(torch_input)

    # output row (8 elements * 2 bytes for bfloat16 = 16 B) is smaller than the 32-byte-rounded
    # CB page size, and L1 is what exposes the writer's over-long NoC write.
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_out = ttnn.operations.moreh.fold(
        tt_input,
        None,
        output_size,
        kernel_size,
        dilation,
        padding,
        stride,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    actual = ttnn.to_torch(tt_out)
    passing, out = comp_allclose_and_pcc(expected, actual, rtol=0.05, atol=0.05)
    logger.info(out)
    assert passing

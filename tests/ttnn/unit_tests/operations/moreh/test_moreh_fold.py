# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc

pytestmark = pytest.mark.use_module_device


def run_moreh_fold_test(input_shape, output_size, kernel_size, dilation, padding, stride, device, memory_config=None):
    # Shifted by 1 so the summed outputs stay away from 0, where bfloat16 rounding error is relatively large.
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16) + 1
    expected = torch.nn.functional.fold(torch_input, output_size, kernel_size, dilation, padding, stride)

    tt_input = ttnn.from_torch(
        torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=memory_config
    )
    tt_output = ttnn.moreh_fold(
        tt_input, None, output_size, kernel_size, dilation, padding, stride, memory_config=memory_config
    )

    passing, output_pcc = comp_allclose_and_pcc(expected, ttnn.to_torch(tt_output), rtol=0.05, atol=0.05)
    assert passing, output_pcc


@pytest.mark.merge_gate
def test_moreh_fold(device):
    torch.manual_seed(0)
    run_moreh_fold_test((1, 9, 32), (6, 10), (3, 3), (1, 1), (0, 0), (1, 1), device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, output_size, kernel_size, dilation, padding, stride",
    [
        ((1, 32, 32), (5, 9), (2, 2), (2, 2), (2, 4), (2, 2)),
        # Input rows of 324 and 36 bfloat16 values are not 32-byte aligned, so DRAM reads go through a scratch buffer.
        ((128, 324), (32, 32), (4, 4), (2, 2), (5, 5), (2, 2)),
        ((5, 64, 36), (12, 12), (4, 4), (2, 2), (3, 3), (2, 2)),
    ],
    ids=["dilation_padding_stride", "rank_2_unaligned_row", "batch_unaligned_row"],
)
def test_moreh_fold_corner_cases(input_shape, output_size, kernel_size, dilation, padding, stride, device):
    torch.manual_seed(0)
    run_moreh_fold_test(input_shape, output_size, kernel_size, dilation, padding, stride, device)


@pytest.mark.merge_gate
def test_moreh_fold_l1_short_row(device):
    torch.manual_seed(0)
    # Regression for #57475: an 8-value (16-byte) output row is shorter than its 32-byte-rounded page, and an
    # L1 output exposed the writer spilling that rounding padding past the row.
    run_moreh_fold_test((1, 4, 21), (4, 8), (2, 2), (1, 1), (0, 0), (1, 1), device, memory_config=ttnn.L1_MEMORY_CONFIG)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "output_size, kernel_size, stride, provide_output, message",
    [
        ([8, 8], [3, 3], [0, 1], False, "stride must be greater than 0"),
        ([8, 8], [0, 3], [1, 1], False, "kernel_size must be greater than 0"),
        # With a provided output the output spec is not computed, so only the validation catches it.
        ([8, 8], [0, 3], [1, 1], True, "kernel_size must be greater than 0"),
        ([8, 8], [3], [1, 1], False, "kernel_size takes 2 elements"),
        ([8], [3, 3], [1, 1], False, "output_size takes 2 elements"),
    ],
    ids=["stride_zero", "kernel_size_zero", "kernel_size_zero_with_output", "kernel_size_arity", "output_size_arity"],
)
def test_moreh_fold_invalid_args(output_size, kernel_size, stride, provide_output, message, device, expect_error):
    torch.manual_seed(0)
    tt_input = ttnn.from_torch(torch.randn(1, 36, 64).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_output = (
        ttnn.from_torch(torch.zeros(1, 4, 8, 8).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        if provide_output
        else None
    )
    with expect_error(RuntimeError, message):
        ttnn.moreh_fold(
            tt_input,
            tt_output,
            output_size=output_size,
            kernel_size=kernel_size,
            dilation=[1, 1],
            padding=[1, 1],
            stride=stride,
        )


@pytest.mark.merge_gate
def test_moreh_fold_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_fold_test((1, 9, 32), (6, 10), (3, 3), (1, 1), (0, 0), (1, 1), device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses.
    tt_placeholder = ttnn.from_torch(torch.zeros(1, 9, 32), device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    run_moreh_fold_test((1, 9, 32), (6, 10), (3, 3), (1, 1), (0, 0), (1, 1), device)
    assert device.num_program_cache_entries() == num_program_cache_entries

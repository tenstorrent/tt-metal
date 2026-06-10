# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal

FLOAT_INPUT_DTYPES = {ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, ttnn.bfloat4_b}


def typecast_test_input_bounds(tt_input_dtype, tt_output_dtype):
    """Out-of-range inputs for integer typecast: clamp (→uint16), wrap (→uint8).

    int32 inputs may be negative; uint16/uint32 inputs stay >= 0 to avoid reinterpretation
    on device (e.g. -1 as uint32 → 4294967295) that would mismatch clamp-aware golden.
    """
    if tt_output_dtype == ttnn.uint16 and tt_input_dtype == ttnn.int32:
        return -1000, 80000
    if tt_output_dtype == ttnn.uint16 and tt_input_dtype == ttnn.uint32:
        return 0, 80000
    if tt_output_dtype == ttnn.uint32 and tt_input_dtype == ttnn.int32:
        return -1000, 80000
    if tt_output_dtype == ttnn.uint8 and tt_input_dtype == ttnn.int32:
        return -512, 512
    if tt_output_dtype == ttnn.uint8 and tt_input_dtype in (ttnn.uint16, ttnn.uint32):
        return 0, 512
    if tt_output_dtype == ttnn.uint8 and tt_input_dtype in FLOAT_INPUT_DTYPES:
        return -512, 512
    if tt_output_dtype == ttnn.uint16 and tt_input_dtype in FLOAT_INPUT_DTYPES:
        return -1000, 80000
    if tt_output_dtype == ttnn.uint32 and tt_input_dtype in FLOAT_INPUT_DTYPES:
        return -1000, 80000
    if tt_output_dtype == ttnn.int32 and tt_input_dtype in FLOAT_INPUT_DTYPES:
        return -1000, 80000
    return 0, 100


def make_typecast_test_input(shape, pt_input_dtype, in_low, in_high):
    if pt_input_dtype == torch.uint8:
        return torch.randint(0, 256, shape, dtype=torch.uint8)
    if pt_input_dtype in (torch.int, torch.int32):
        return torch.randint(in_low, in_high + 1, shape, dtype=torch.int32)
    return (torch.rand(shape) * (in_high - in_low) + in_low).to(pt_input_dtype)


def assert_integer_typecast_equal(expected, actual):
    assert_equal(expected.to(torch.int64), actual.to(torch.int64))


def uses_exact_integer_typecast_check(tt_input_dtype, tt_output_dtype):
    if tt_output_dtype in (ttnn.int32, ttnn.uint32, ttnn.uint8):
        return True
    return tt_output_dtype == ttnn.uint16 and tt_input_dtype in (
        ttnn.int32,
        ttnn.uint16,
        ttnn.uint32,
        ttnn.uint8,
    )

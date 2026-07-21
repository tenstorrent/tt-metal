# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal

INTEGER_OUTPUT_DTYPES = frozenset({ttnn.uint8, ttnn.uint16, ttnn.uint32, ttnn.int32})
_UNSIGNED_INPUT_DTYPES = frozenset({ttnn.uint8, ttnn.uint16, ttnn.uint32})

# Signed 32-bit extrema: min = -2^(32-1), max = 2^(32-1) - 1 (exponent is 31, not 32).
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1

_UNSIGNED_INPUT_MAX = {
    ttnn.uint8: 255,
    ttnn.uint16: 65535,
    # uint32 inputs are stored as int32 in tests; clamp_high fits without wrap.
    ttnn.uint32: _INT32_MAX,
}
# Representable limits per dtype (uint32 output capped to uint16-scale for practical randint).
_DTYPE_MIN = {
    ttnn.uint8: 0,
    ttnn.uint16: 0,
    ttnn.uint32: 0,
    ttnn.int32: _INT32_MIN,
}
_DTYPE_MAX = {
    ttnn.uint8: 255,
    ttnn.uint16: 65535,
    ttnn.uint32: 65535,
    ttnn.int32: _INT32_MAX,
}

# Extend ~14465 above uint16 max → 80000 (clamp headroom without full uint32 range).
_CLAMP_HIGH = _DTYPE_MAX[ttnn.uint16] + 14465
# Practical negative magnitude for signed/float inputs (beyond output min, randint-friendly).
_CLAMP_LOW_MAGNITUDE = 1000


def _output_allows_negative(tt_output_dtype):
    return tt_output_dtype == ttnn.int32


def typecast_test_input_bounds(tt_input_dtype, tt_output_dtype):
    """Derive out-of-range input bounds from dtype limits for integer typecast tests.

    Extends below/above the output dtype range so clamp (→uint16), wrap (→uint8), and
    saturation paths are exercised. Unsigned inputs stay >= 0 to avoid reinterpretation
    on device (e.g. -1 as uint32 → 4294967295) that would mismatch clamp-aware golden.

    Rules (per review on #46574):
    - Unsigned input → low = 0.
    - Signed/float input + uint8 output → ±2×(max+1) for wrap/clamp (±512).
    - Signed/float input + wider unsigned/signed output → low = -min(1000, out_max+1),
      high = out_max + slack (80000 for uint16-scale outputs).
    - Unsigned input + uint8 output → [0, min(512, input_max)]; wider outputs → [0, min(80000, input_max)].
    """
    if tt_output_dtype not in INTEGER_OUTPUT_DTYPES:
        return 0, 100

    out_min = _DTYPE_MIN[tt_output_dtype]
    out_max = _DTYPE_MAX[tt_output_dtype]

    if tt_input_dtype in _UNSIGNED_INPUT_DTYPES:
        low = 0
    elif tt_output_dtype == ttnn.uint8:
        span = out_max + 1
        low = -2 * span
    elif not _output_allows_negative(tt_output_dtype):
        low = -min(_CLAMP_LOW_MAGNITUDE, out_max + 1)
    else:
        low = -min(_CLAMP_LOW_MAGNITUDE, abs(out_min) if out_min < 0 else _CLAMP_LOW_MAGNITUDE)

    if tt_output_dtype == ttnn.uint8:
        high = 2 * (out_max + 1)
    elif tt_output_dtype in (ttnn.uint16, ttnn.uint32, ttnn.int32):
        high = _CLAMP_HIGH
    else:
        high = out_max + 100

    # Unsigned inputs cannot represent values above their dtype max (wrap on device).
    if tt_input_dtype in _UNSIGNED_INPUT_DTYPES:
        high = min(high, _UNSIGNED_INPUT_MAX[tt_input_dtype])

    return low, high


def make_typecast_test_input(shape, pt_input_dtype, in_low, in_high):
    if pt_input_dtype == torch.uint8:
        return torch.randint(0, 256, shape, dtype=torch.uint8)
    if pt_input_dtype in (torch.int, torch.int32):
        return torch.randint(in_low, in_high + 1, shape, dtype=torch.int32)
    return (torch.rand(shape) * (in_high - in_low) + in_low).to(pt_input_dtype)


def assert_integer_typecast_equal(expected, actual):
    assert_equal(expected.to(torch.int64), actual.to(torch.int64))


def uses_exact_integer_typecast_check(_tt_input_dtype, tt_output_dtype):
    return tt_output_dtype in INTEGER_OUTPUT_DTYPES

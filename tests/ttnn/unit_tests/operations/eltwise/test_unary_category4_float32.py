# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_allclose, assert_equal
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    assert_ftz_then_ulp,
    generate_float32_bits,
    to_float32_tt_tensor,
    SMALLEST_NORMAL_BF16,
)

pytestmark = pytest.mark.use_module_device


"""
Category 4: ops with a float scalar parameter

| #  | Op              | Parameter           | Formula                                     |
|----|-----------------|---------------------|---------------------------------------------|
| 1  | ttnn.elu        | alpha               | x > 0 ? x : alpha * (exp(x) - 1)            |
| 2  | ttnn.heaviside  | value               | x < 0 ? 0 : x > 0 ? 1 : value               |
| 3  | ttnn.leaky_relu | negative_slope      | x >= 0 ? x : negative_slope * x             |
| 4  | ttnn.relu_max   | upper_limit         | max(0, min(x, upper_limit))                 |
| 5  | ttnn.relu_min   | lower_limit         | max(x, lower_limit)                         |
| 6  | ttnn.rpow       | exponent            | exponent ** x                               |
| 7  | ttnn.celu       | alpha               | x > 0 ? x : alpha * (exp(x / alpha) - 1)    |
| 8  | ttnn.fill       | fill_value          | fill_value (input ignored)                  |
| 9  | ttnn.hardshrink | lambd               | |x| > lambd ? x : 0                         |
| 10 | ttnn.softshrink | lambd               | x > lambd ? x - lambd : x < -lambd ?        |
|    |                 |                     | x + lambd : 0                               |

Every op except fill sweeps all normal bfloat16 values through the tensor
input, with the scalar parameter parametrized over a representative set. fill
ignores its input, so fill_value itself is the swept domain.

softcap is not covered here: it does not support float32 inputs.

Accuracy criteria (FTZ = flush-to-zero, SNB = smallest normal bfloat16)

| Op                    | Parameter            | Input range        | Check                       |
|-----------------------|----------------------|--------------------|-----------------------------|
| heaviside, relu_max,  | all                  | all                | exact                       |
| relu_min, hardshrink  |                      |                    |                             |
| fill                  | all                  | n/a                | exact (bit pattern)         |
| softshrink            | all                  | all                | exact                       |
| leaky_relu            | all                  | exact result < SNB | exactly 0                   |
|                       |                      | elsewhere          | ULP ≤ 1                     |
| elu                   | alpha = ±2^k or 0    | exact result < SNB | exactly 0                   |
|                       |                      | elsewhere          | ULP ≤ 1                     |
|                       | alpha = 0.1          | elsewhere          | ULP ≤ 2                     |
| celu                  | alpha > 1            | -alpha*SNB < x < 0 | exactly 0                   |
|                       | alpha = 2^k          | elsewhere          | ULP ≤ 1                     |
|                       | alpha = 0.1          | elsewhere          | ULP ≤ 2                     |
| rpow                  | all                  | |x| >= 2^116       | xfail (#55129)              |
|                       |                      | exact result < SNB | exactly 0                   |
|                       |                      | overflow           | +inf                        |
|                       | exponent = 2^k       | elsewhere          | ULP ≤ 1                     |
|                       | other exponents      | elsewhere          | rtol 1e-5 (#59761)          |
"""


# ─────────────────────────────────────────────────────────────────────────────
# heaviside, relu_max, relu_min, hardshrink — exact (comparison + select)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0, -1.0, 2.5])
def test_heaviside(device, value):
    """heaviside(x, value) = 0 if x < 0, value if x == 0, 1 if x > 0.

    | Input range | Check | Reason                               |
    |-------------|-------|--------------------------------------|
    | all         | exact | every output is 0, 1 or value        |
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.heaviside)
    golden = golden_function(input_tensor, value, device=device)

    result = ttnn.to_torch(ttnn.heaviside(tt_in, value))
    assert_equal(golden, result)


@pytest.mark.parametrize("upper_limit", [0.0, 1.0, 3.0, 100.0, -1.0])
def test_relu_max(device, upper_limit):
    """relu_max(x, upper_limit) = max(0, min(x, upper_limit)).

    | Input range | Check | Reason                                         |
    |-------------|-------|------------------------------------------------|
    | all         | exact | every output is x, 0 or upper_limit            |

    upper_limit = -1 collapses every output to 0.
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.relu_max)
    golden = golden_function(input_tensor, upper_limit, device=device)

    result = ttnn.to_torch(ttnn.relu_max(tt_in, upper_limit))
    assert_equal(golden, result)


@pytest.mark.parametrize("lower_limit", [0.0, -1.0, -3.0, -100.0, 1.0])
def test_relu_min(device, lower_limit):
    """relu_min(x, lower_limit) = max(x, lower_limit).

    | Input range | Check | Reason                                 |
    |-------------|-------|----------------------------------------|
    | all         | exact | every output is x or lower_limit       |

    lower_limit = 1 puts the bound above 0.
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.relu_min)
    golden = golden_function(input_tensor, lower_limit, device=device)

    result = ttnn.to_torch(ttnn.relu_min(tt_in, lower_limit))
    assert_equal(golden, result)


@pytest.mark.parametrize("lambd", [0.5, 0.25, 1.0, 2.0, 4.0])
def test_hardshrink(device, lambd):
    """hardshrink(x, lambd) = x if |x| > lambd else 0.

    | Input range | Check | Reason                         |
    |-------------|-------|--------------------------------|
    | all         | exact | every output is x or 0         |
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.hardshrink)
    golden = golden_function(input_tensor, lambd=lambd, device=device)

    result = ttnn.to_torch(ttnn.hardshrink(tt_in, lambd=lambd))
    assert_equal(golden, result)


# ─────────────────────────────────────────────────────────────────────────────
# softshrink, leaky_relu — one add or multiply on the non-identity branch
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("lambd", [0.5, 0.25, 1.0, 2.0])
def test_softshrink(device, lambd):
    """softshrink(x, lambd) = x - lambd if x > lambd, x + lambd if x < -lambd, else 0.

    | Input range | Check | Reason                                          |
    |-------------|-------|-------------------------------------------------|
    | all         | exact | one correctly rounded add, same as the golden   |
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.softshrink)
    golden = golden_function(input_tensor, lambd=lambd, device=device)

    result = ttnn.to_torch(ttnn.softshrink(tt_in, lambd=lambd))
    assert_equal(golden, result)


@pytest.mark.parametrize("negative_slope", [0.01, 0.1, 1.0, 2.0, -0.5])
def test_leaky_relu(device, negative_slope):
    """leaky_relu(x, negative_slope) = x if x >= 0, else negative_slope * x.

    | Input range        | Check     | Reason                                     |
    |--------------------|-----------|--------------------------------------------|
    | exact result < SNB | exactly 0 | negative_slope * x is subnormal; FTZ       |
    | elsewhere          | ULP ≤ 1   | one multiply (exact for slope = ±2^k)      |

    |negative_slope| > 1 overflows the most negative inputs to ±inf on both sides.
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.leaky_relu)
    golden = golden_function(input_tensor, negative_slope=negative_slope, device=device)

    x64 = input_tensor.double()
    exact = torch.where(x64 >= 0, x64, x64 * negative_slope)

    result = ttnn.to_torch(ttnn.leaky_relu(tt_in, negative_slope=negative_slope))
    assert_ftz_then_ulp(golden, result, ulp_threshold=1, reference=exact, allow_nonfinite=True)


# ─────────────────────────────────────────────────────────────────────────────
# elu, celu — alpha * expm1 on the negative branch
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "alpha, ulp_threshold",
    [(1.0, 1), (0.5, 1), (2.0, 1), (0.1, 2), (0.0, 1), (-1.0, 1), (-0.5, 1)],
)
def test_elu(device, alpha, ulp_threshold):
    """elu(x, alpha) = x if x > 0, else alpha * (exp(x) - 1).

    | alpha        | Input range        | Check     | Reason                                    |
    |--------------|--------------------|-----------|-------------------------------------------|
    | all          | exact result < SNB | exactly 0 | alpha * expm1(x) is subnormal; FTZ        |
    | ±2^k, 0      | elsewhere          | ULP ≤ 1   | expm1 error; scaling by 2^k is exact      |
    | 0.1          | elsewhere          | ULP ≤ 2   | the alpha multiply adds a rounding (2 ULP |
    |              |                    |           | at x = -0.0088, -0.066, -0.357)           |

    No cancellation band is needed near 0: the whole lattice meets the bound.
    alpha = 0 and negative alpha are valid for elu, unlike celu.
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.elu)
    golden = golden_function(input_tensor, alpha=alpha, device=device)

    x64 = input_tensor.double()
    exact = torch.where(x64 > 0, x64, alpha * torch.expm1(x64))

    result = ttnn.to_torch(ttnn.elu(tt_in, alpha=alpha))
    assert_ftz_then_ulp(golden, result, ulp_threshold, reference=exact, allow_nonfinite=True)


@pytest.mark.parametrize("alpha, ulp_threshold", [(1.0, 1), (0.5, 1), (2.0, 1), (0.1, 2)])
def test_celu(device, alpha, ulp_threshold):
    """celu(x, alpha) = x if x > 0, else alpha * (exp(x / alpha) - 1).

    | alpha  | Input range        | Check     | Reason                                     |
    |--------|--------------------|-----------|--------------------------------------------|
    | > 1    | -alpha * SNB < x   | exactly 0 | x / alpha is subnormal and FTZ'd, although |
    |        | < 0                |           | the result ≈ x is normal                   |
    | all    | exact result < SNB | exactly 0 | FTZ                                        |
    | 2^k    | elsewhere          | ULP ≤ 1   | expm1 error; x / 2^k and * 2^k are exact   |
    | 0.1    | elsewhere          | ULP ≤ 2   | x / alpha and * alpha each round (2 ULP at |
    |        |                    |           | x = -2.3e-7, -0.0327)                      |
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.celu)
    golden = golden_function(input_tensor, alpha=alpha, device=device)

    x64 = input_tensor.double()
    exact = torch.where(x64 > 0, x64, alpha * torch.expm1(x64 / alpha))

    result = ttnn.to_torch(ttnn.celu(tt_in, alpha=alpha))

    scaled_ftz = (input_tensor < 0) & (input_tensor > -alpha * SMALLEST_NORMAL_BF16)
    if alpha > 1:
        assert scaled_ftz.any(), "expected the -alpha * SNB < x < 0 band to be non-empty for alpha > 1"
    assert_equal(torch.zeros_like(result[scaled_ftz]), result[scaled_ftz])

    keep = ~scaled_ftz
    assert_ftz_then_ulp(golden[keep], result[keep], ulp_threshold, reference=exact[keep], allow_nonfinite=True)


# ─────────────────────────────────────────────────────────────────────────────
# rpow — exponent ** x via 2 ** (x * log2(exponent))
# ─────────────────────────────────────────────────────────────────────────────

# The float32 kernel splits x with x * 4097 (a Veltkamp split), which overflows
# to inf for |x| >= FLT_MAX / 4097, just above 2^116.
RPOW_SPLIT_OVERFLOW = 2.0**116

RPOW_EXPONENTS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 8.0, 9.0, 10.0]


@pytest.mark.parametrize("exponent", RPOW_EXPONENTS)
def test_rpow(device, exponent):
    """rpow(x, exponent) = exponent ** x, for |x| < 2^116.

    | exponent | Input range        | Check     | Reason                                     |
    |----------|--------------------|-----------|--------------------------------------------|
    | all      | exact result < SNB | exactly 0 | underflow; FTZ                             |
    |          | overflow           | ±inf      | non-finite values must match the golden    |
    | 2^k      | elsewhere          | ULP ≤ 1   | log2(exponent) is exact                    |
    | other    | elsewhere          | rtol 1e-5 | the rounding of log2(exponent) is scaled   |
    |          |                    |           | by x: up to 70 ULP (exponent = 3, x = 78); |
    |          |                    |           | #59761                                     |

    |x| >= 2^116 is test_rpow_split_overflow.
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.rpow)
    golden = golden_function(input_tensor, exponent, device=device)

    result = ttnn.to_torch(ttnn.rpow(tt_in, exponent))

    in_range = input_tensor.abs() < RPOW_SPLIT_OVERFLOW
    golden, result, input_tensor = golden[in_range], result[in_range], input_tensor[in_range]
    exact = torch.pow(torch.tensor(exponent, dtype=torch.float64), input_tensor.double())

    if math.log2(exponent).is_integer():
        assert_ftz_then_ulp(golden, result, ulp_threshold=1, reference=exact, allow_nonfinite=True)
        return

    ftz = exact.abs() < SMALLEST_NORMAL_BF16
    assert_equal(torch.zeros_like(result[ftz]), result[ftz])
    golden, result = golden[~ftz], result[~ftz]
    overflow = torch.isinf(golden)
    assert torch.equal(overflow, torch.isinf(result)), "rpow overflow positions differ from the golden"
    assert torch.equal(golden[overflow], result[overflow]), "rpow overflow signs differ from the golden"
    finite = torch.isfinite(golden)
    assert_allclose(expected_result=golden[finite], actual_result=result[finite], rtol=1e-5, atol=0)


@pytest.mark.parametrize("exponent", [0.5, 1.0, 2.0])
@pytest.mark.xfail(strict=True, reason="#55129: float32 pow overflows its Veltkamp split (x * 4097) for |x| >= 2^116")
def test_rpow_split_overflow(device, exponent):
    """rpow(x, exponent) for |x| >= 2^116.

    | Input range  | Check           | Device returns                            |
    |--------------|-----------------|-------------------------------------------|
    | |x| >= 2^116 | matches golden: | 0 where the golden is +inf, +inf where it |
    |              | 0, 1 or +inf    | is 0, and 0 or +inf where it is 1         |
    """
    input_tensor = generate_float32_bits()
    input_tensor = input_tensor[input_tensor.abs() >= RPOW_SPLIT_OVERFLOW].reshape(1, -1)

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.rpow)
    golden = golden_function(input_tensor, exponent, device=device)

    result = ttnn.to_torch(ttnn.rpow(tt_in, exponent))
    assert_equal(golden, result)


# ─────────────────────────────────────────────────────────────────────────────
# fill — exact (bit pattern); fill_value is the swept domain
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fill_value",
    [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        100.0,
        -100.0,
        torch.finfo(torch.float32).tiny,
        -torch.finfo(torch.float32).tiny,
        torch.finfo(torch.float32).max,
        -torch.finfo(torch.float32).max,
        0.1,
        -10.0,
        15.5,
        -29.5,
        2147483647.0,
        -2147483648.0,
    ],
)
def test_fill(device, fill_value):
    """fill(x, fill_value) = fill_value everywhere; x is ignored.

    | fill_value | Check               | Reason                                         |
    |------------|---------------------|------------------------------------------------|
    | all        | exact (bit pattern) | the scalar's float32 bits are written as-is,   |
    |            |                     | including the sign of -0.0                     |

    0.1 is not a dyadic rational, so both sides must use the same float32
    rounding of the Python float.
    """
    input_tensor = torch.arange(256 * 256, dtype=torch.float32).view(256, 256)

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.fill)
    golden = golden_function(input_tensor, fill_value, device=device)

    result = ttnn.to_torch(ttnn.fill(tt_in, fill_value))

    assert torch.equal(result.view(torch.int32), golden.view(torch.int32)), (
        f"fill(fill_value={fill_value}) diverged for "
        f"{int((result.view(torch.int32) != golden.view(torch.int32)).sum().item())} of {result.numel()} elements"
    )


@pytest.mark.parametrize("fill_values", [(0.0, -0.0, 0.0), (-0.0, 0.0, -0.0)], ids=["pos_first", "neg_first"])
def test_fill_signed_zero_program_cache(device, fill_values):
    """+0.0 and -0.0 fill values must not share a cached program (#62557).

    | Check               | Reason                                                  |
    |---------------------|---------------------------------------------------------|
    | exact (bit pattern) | fill_value's bits are embedded in the kernel defines    |

    Each sign compiles its own program on first use and hits it on reuse, so the
    output does not depend on which sign ran first.
    """
    tt_in = to_float32_tt_tensor(torch.ones((64, 64), dtype=torch.float32), device)
    device.clear_program_cache()

    seen_signs = set()
    prev_entries = device.num_program_cache_entries()
    for fill_value in fill_values:
        expected = torch.full((64, 64), fill_value, dtype=torch.float32)
        tt_out = ttnn.fill(tt_in, fill_value)
        entries = device.num_program_cache_entries()
        sign = math.copysign(1.0, fill_value)
        if sign in seen_signs:
            assert entries == prev_entries, f"fill_value {fill_value} should hit its cached program"
        else:
            assert entries > prev_entries, f"fill_value {fill_value} should compile a new program"
            seen_signs.add(sign)
        prev_entries = entries
        result = ttnn.to_torch(tt_out)
        assert torch.equal(
            result.view(torch.int32), expected.view(torch.int32)
        ), f"fill(fill_value={fill_value}) returned the wrong sign of zero"

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import torch.nn.functional as F
import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_ulp, assert_with_pcc, assert_allclose
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    assert_all_nonfinite,
    assert_finite_matches_golden,
    bf16_binade_samples,
    generate_float32_bits,
    generate_float32_bits_in_range,
    to_float32_tt_tensor,
    SMALLEST_NORMAL_BF16,
    MAX_BF16,
)

pytestmark = pytest.mark.use_module_device


"""
Float32 accuracy sweep of the same ops and input ranges as
test_unary_category3_bfloat16.py.

Inputs are the bfloat16 lattice stored as float32
(generate_float32_bits / generate_float32_bits_in_range): all 65,536
bfloat16 encodings, promoted without rounding. The op runs at ttnn.float32
and ULP thresholds below are float32 ULPs.


Category 3: ops with a fast_and_approximate_mode parameter

| #  | Op         | Description                     |
|----|------------|---------------------------------|
| 1  | ttnn.sqrt  | Square root                     |
| 2  | ttnn.rsqrt | Reciprocal square root          |
| 3  | ttnn.exp   | Exponential                     |
| 4  | ttnn.erf   | Error function                  |
| 5  | ttnn.gelu  | Gaussian error linear unit      |
| 6  | ttnn.log   | Natural logarithm               |
| 7  | ttnn.log2  | Base-2 logarithm                |
| 8  | ttnn.log10 | Base-10 logarithm               |
| 9  | ttnn.log1p | log(1 + x)                      |
| 10 | ttnn.mish  | Mish activation                 |

Each op runs in accurate mode (fast_and_approximate_mode=False) and fast mode
(True). Thresholds come from the tests these replaced (test_math, test_unary,
test_activation, test_unary_ops_ttnn, test_unary_category1).

Accuracy criteria (SNB = smallest normal bfloat16)

| Op               | Mode     | Input range           | Check                        |
|------------------|----------|-----------------------|------------------------------|
| sqrt             | accurate | [0, MAX_BF16]         | ULP ≤ 1                      |
|                  | accurate | x < 0                 | non-finite                   |
| rsqrt            | accurate | [SNB, MAX_BF16]       | ULP ≤ 2                      |
|                  | accurate | x <= 0                | non-finite                   |
| sqrt, rsqrt      | fast     | [1, 100]              | ULP ≤ 2                      |
| exp              | fast     | [-87, 88.5]           | PCC ≥ 0.999, rtol 0.05,      |
|                  |          |                       | atol 1e-6                    |
|                  | both     | x <= -87.5            | exactly 0                    |
|                  | both     | x >= 89               | +inf (fast mode xfail: NaN)  |
| erf              | accurate | [-10, 10]             | ULP ≤ 3 (BH 4)               |
|                  | fast     | [-10, 10]             | rtol 1.1e-4 (BH 5.4e-3)      |
| gelu             | accurate | x >= 0                | ULP ≤ 2                      |
|                  | accurate | -1 < x < 0            | ULP ≤ 3                      |
|                  | accurate | x <= -1               | abs ≤ 4.5e-7                 |
|                  | fast     | [-10, 10]             | PCC ≥ 0.999, rtol/atol 0.05  |
| log, log2        | accurate | [SNB, MAX_BF16]       | ULP ≤ 1                      |
| log10            | accurate | [SNB, MAX_BF16]       | ULP ≤ 2                      |
| log, log2, log10 | both     | x <= 0                | non-finite                   |
| log1p            | accurate | x > -1                | ULP ≤ 1                      |
|                  | accurate | x <= -1               | non-finite                   |
| log family       | fast     | [1, 100]              | atol 0.0625                  |
| mish             | both     | all normals           | rtol 1e-5, atol 0.02         |

assert_allclose scales rtol by the device output, not the golden as its
docstring says. This is a quirk of the shared helper.
"""


# ─────────────────────────────────────────────────────────────────────────────
# sqrt, rsqrt — reciprocal/root ops, ULP-based on their positive domain
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op, golden_fn, low, ulp_accurate",
    [
        (ttnn.sqrt, torch.sqrt, 0.0, 1),
        (ttnn.rsqrt, torch.rsqrt, SMALLEST_NORMAL_BF16, 2),
    ],
    ids=["sqrt", "rsqrt"],
)
def test_root_ops_accurate(device, ttnn_op, golden_fn, low, ulp_accurate):
    """Float32 sweep over the non-negative bfloat16 values for sqrt and rsqrt, accurate mode.

    | Op    | Input range     | Check   |
    |-------|-----------------|---------|
    | sqrt  | [0, MAX_BF16]   | ULP ≤ 1 |
    | rsqrt | [SNB, MAX_BF16] | ULP ≤ 2 |

    rsqrt leaves out 0, since rsqrt(0) = +inf (see test_root_ops_negative_domain).
    """
    input_tensor = generate_float32_bits_in_range(low, MAX_BF16)
    tt_in = to_float32_tt_tensor(input_tensor, device)

    golden = golden_fn(input_tensor.float()).to(torch.float32)
    result = ttnn.to_torch(ttnn_op(tt_in, fast_and_approximate_mode=False)).to(torch.float32)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=ulp_accurate, allow_nonfinite=True)


@pytest.mark.parametrize(
    "ttnn_op, golden_fn",
    [
        (ttnn.sqrt, torch.sqrt),
        (ttnn.rsqrt, torch.rsqrt),
    ],
    ids=["sqrt", "rsqrt"],
)
def test_root_ops_fast(device, ttnn_op, golden_fn):
    """sqrt and rsqrt in fast mode.

    | Input range | Check   | Reason                                 |
    |-------------|---------|----------------------------------------|
    | [1, 100]    | ULP ≤ 2 | same bound as test_unary_root_ops_ttnn |
    """
    input_tensor = generate_float32_bits_in_range(1.0, 100.0)
    tt_in = to_float32_tt_tensor(input_tensor, device)

    golden = golden_fn(input_tensor.float()).to(torch.float32)
    result = ttnn.to_torch(ttnn_op(tt_in, fast_and_approximate_mode=True)).to(torch.float32)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=2)


@pytest.mark.parametrize(
    "ttnn_op, high",
    [
        (ttnn.sqrt, -SMALLEST_NORMAL_BF16),  # sqrt(0) = 0 is in-domain; keep 0 out of this sweep
        (ttnn.rsqrt, 0.0),  # rsqrt(0) = +inf is out-of-domain; include 0 here
    ],
    ids=["sqrt", "rsqrt"],
)
def test_root_ops_negative_domain(device, ttnn_op, high):
    """sqrt and rsqrt outside their domain, accurate mode.

    | Op    | Input range | Check      | Reason                          |
    |-------|-------------|------------|---------------------------------|
    | sqrt  | x < 0       | non-finite | sqrt(0) = 0 is in the domain    |
    | rsqrt | x <= 0      | non-finite | rsqrt(0) = +inf                 |

    Fast mode is not tested here. It doesn't check the domain and returns
    finite values for negative inputs instead of NaN or inf.
    """
    input_tensor = generate_float32_bits_in_range(-MAX_BF16, high)
    tt_in = to_float32_tt_tensor(input_tensor, device)

    result = ttnn.to_torch(ttnn_op(tt_in, fast_and_approximate_mode=False))
    assert_all_nonfinite(result, f"{ttnn_op.__name__}(x<=0)" if high == 0.0 else f"{ttnn_op.__name__}(x<0)")


# ─────────────────────────────────────────────────────────────────────────────
# exp — fast mode over the finite range, plus the underflow and overflow tails in both modes
# (accurate mode over the finite range is test_exp_ops in test_unary_category1_float32.py)
# ─────────────────────────────────────────────────────────────────────────────


def test_exp_fast(device):
    """exp in fast mode over its finite range.

    | Input range | Check                     | Reason                                 |
    |-------------|---------------------------|----------------------------------------|
    | [-87, 88.5] | finite where golden is    | PCC can hide a stray NaN or inf        |
    |             | PCC ≥ 0.999               | weak alone: exp spans ~76 decades, so  |
    |             |                           | the largest values dominate            |
    |             | rtol 0.05, atol 1e-6      | the real accuracy gate                 |

    Below x ≈ -13.8 the golden is under atol, so the allclose bound no longer
    says much. Fast mode's decay toward 0 there is not checked.
    """
    input_tensor = generate_float32_bits_in_range(-87.0, 88.5)
    tt_in = to_float32_tt_tensor(input_tensor, device)

    golden = torch.exp(input_tensor.float()).to(torch.float32)
    result = ttnn.to_torch(ttnn.exp(tt_in, fast_and_approximate_mode=True)).to(torch.float32)

    assert_finite_matches_golden(golden, result, "exp(fast)")
    assert_with_pcc(golden, result, pcc=0.999)
    assert_allclose(expected_result=golden, actual_result=result, rtol=0.05, atol=1e-6)


@pytest.mark.parametrize("fast", [False, True], ids=["accurate", "fast"])
def test_exp_underflow(device, fast):
    """exp underflow tail, both modes, sampled per binade.

    | Input range                                     | Check     |
    |-------------------------------------------------|-----------|
    | [-MAX_BF16, -87.5]: first, middle and last      | exactly 0 |
    | bfloat16 mantissa of every binade, plus -87.5   |           |

    Every input here flushes to 0, so a few values per binade (power-of-two
    interval) are enough to catch a magnitude where range reduction breaks.
    """
    input_tensor = -bf16_binade_samples(87.5)
    tt_in = to_float32_tt_tensor(input_tensor, device)

    result = ttnn.to_torch(ttnn.exp(tt_in, fast_and_approximate_mode=fast))
    assert torch.all(result == 0.0), "expected exp underflow region (x <= -87.5) to be exactly 0"


@pytest.mark.parametrize(
    "fast",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                strict=True,
                reason="#60792: fast-mode float32 exp returns NaN instead of +inf for x >= 89",
            ),
        ),
    ],
    ids=["accurate", "fast"],
)
def test_exp_overflow(device, fast):
    """exp overflow tail, both modes, sampled per binade.

    | Input range                                     | Check |
    |-------------------------------------------------|-------|
    | [89, MAX_BF16]: first, middle and last bfloat16 | +inf  |
    | mantissa of every binade, plus 89               |       |

    Every input here overflows to +inf, so a few values per binade are enough.
    Fast mode returns NaN for every input here.
    """
    input_tensor = bf16_binade_samples(89.0)
    tt_in = to_float32_tt_tensor(input_tensor, device)

    result = ttnn.to_torch(ttnn.exp(tt_in, fast_and_approximate_mode=fast))
    assert torch.all(torch.isposinf(result)), "expected exp overflow region (x > 88.5) to be +inf"


# ─────────────────────────────────────────────────────────────────────────────
# erf — bounded [-1, 1]; ULP ≤ 3 (BH 4) in accurate mode, rtol 1.1e-4 (BH 5.4e-3) in fast mode
# ─────────────────────────────────────────────────────────────────────────────


def test_erf_accurate(device):
    """erf in accurate mode.

    | Arch      | Input range | Check   | Measured                                       |
    |-----------|-------------|---------|------------------------------------------------|
    | wormhole  | [-10, 10]   | ULP ≤ 3 | max 3 ULP at 68 values, spread over the range  |
    | blackhole | [-10, 10]   | ULP ≤ 4 | max 4 ULP at 10 values: ±0.2119, ±0.2139,      |
    |           |             |         | ±0.2236, ±0.4570, ±1.8516; device is slightly  |
    |           |             |         | smaller in magnitude each time                 |
    """
    input_tensor = generate_float32_bits_in_range(-10.0, 10.0)
    tt_in = to_float32_tt_tensor(input_tensor, device)

    golden = torch.erf(input_tensor.float()).to(torch.float32)
    result = ttnn.to_torch(ttnn.erf(tt_in, fast_and_approximate_mode=False)).to(torch.float32)

    ulp_threshold = 4 if is_blackhole() else 3
    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=ulp_threshold)


def test_erf_fast(device):
    """erf in fast mode. The relative error is near-constant, so ULP doesn't fit.

    | Arch      | Input range | Check               | Measured max relative error            |
    |-----------|-------------|---------------------|----------------------------------------|
    | wormhole  | [-10, 10]   | rtol 1.1e-4, atol 0 | 1.02e-4 (up to 1711 ULP)               |
    | blackhole | [-10, 10]   | rtol 5.4e-3, atol 0 | 5.35e-3 at x = ±6.03, where the device |
    |           |             |                     | saturates to ±0.9947 instead of ±1     |

    Blackhole is roughly bfloat16-accurate in fast mode: median relative error
    2^-8, about 50 times worse than wormhole.
    """
    input_tensor = generate_float32_bits_in_range(-10.0, 10.0)
    tt_in = to_float32_tt_tensor(input_tensor, device)

    golden = torch.erf(input_tensor.float()).to(torch.float32)
    result = ttnn.to_torch(ttnn.erf(tt_in, fast_and_approximate_mode=True)).to(torch.float32)

    rtol = 5.4e-3 if is_blackhole() else 1.1e-4
    assert_allclose(expected_result=golden, actual_result=result, rtol=rtol, atol=0)


# ─────────────────────────────────────────────────────────────────────────────
# gelu — accurate exhaustive (ULP for x > -1, abs below); fast PCC+allclose
# ─────────────────────────────────────────────────────────────────────────────


def test_gelu_accurate(device):
    """Float32 sweep over every normal bfloat16 value for gelu, accurate mode.

    | Input range            | Check         | Reason                                     |
    |------------------------|---------------|--------------------------------------------|
    | 2^-126 <= |x| < 2^-125 | not checked   | x/2 is subnormal and the device flushes it |
    | x >= 0                 | ULP ≤ 2       | measured max 2                             |
    | -1 < x < 0             | ULP ≤ 3       | measured max 3                             |
    | x <= -1                | abs ≤ 4.5e-7  | ~2-4e-7 absolute error (max 4.44e-7 near   |
    |                        |               | x = -5.06) as the result shrinks toward 0; |
    |                        |               | ULP grows to 11 at -1.36, 595 at -2.95     |
    | finite golden, outside | finite        | a stray NaN or inf must not slip through   |
    | the flush band         |               |                                            |

    The abs bound stops saying much below x ≈ -5.5, where |result| < 1e-7.

    The golden is computed in float64, because fp32's 1 + erf(x/sqrt(2))
    cancels for |x| around 5-6.
    """
    input_tensor = generate_float32_bits()  # all normals; specials→0
    tt_in = to_float32_tt_tensor(input_tensor, device)

    golden = F.gelu(input_tensor.double()).to(torch.float32)
    result = ttnn.to_torch(ttnn.gelu(tt_in, fast_and_approximate_mode=False)).to(torch.float32)

    abs_x = input_tensor.abs().float()
    exp1_ftz_band = (abs_x >= 2.0**-126) & (abs_x < 2.0**-125)
    assert exp1_ftz_band.any(), "expected exp-field=1 FTZ band to be non-empty for this exhaustive sweep"

    finite_golden = torch.isfinite(golden)
    unexpected_nonfinite = finite_golden & ~exp1_ftz_band & ~torch.isfinite(result)
    assert not unexpected_nonfinite.any(), (
        f"gelu(accurate): {int(unexpected_nonfinite.sum())} finite-input elements outside the documented "
        "FTZ band produced a non-finite device output"
    )

    checked = ~exp1_ftz_band & finite_golden
    x = input_tensor.float()
    non_negative = checked & (x >= 0)
    small_negative = checked & (x < 0) & (x > -1)
    large_negative = checked & (x <= -1)
    assert non_negative.any() and small_negative.any() and large_negative.any()

    assert_with_ulp(expected_result=golden[non_negative], actual_result=result[non_negative], ulp_threshold=2)
    assert_with_ulp(expected_result=golden[small_negative], actual_result=result[small_negative], ulp_threshold=3)
    assert_allclose(expected_result=golden[large_negative], actual_result=result[large_negative], rtol=0, atol=4.5e-7)


def test_gelu_fast(device):
    """gelu in fast mode (lookup table).

    | Input range | Check                  | Reason                                     |
    |-------------|------------------------|--------------------------------------------|
    | [-10, 10]   | finite where golden is | PCC can hide a stray NaN or inf            |
    |             | PCC ≥ 0.999            | weak alone: the near-identity large-|x|    |
    |             |                        | tail dominates                             |
    |             | rtol 0.05, atol 0.05   | the real accuracy gate                     |

    The golden is computed in float64, as in test_gelu_accurate.
    """
    input_tensor = generate_float32_bits_in_range(-10.0, 10.0)
    tt_in = to_float32_tt_tensor(input_tensor, device)

    golden = F.gelu(input_tensor.double()).to(torch.float32)
    result = ttnn.to_torch(ttnn.gelu(tt_in, fast_and_approximate_mode=True)).to(torch.float32)

    assert_finite_matches_golden(golden, result, "gelu(fast)")
    assert_with_pcc(golden, result, pcc=0.999)
    assert_allclose(expected_result=golden, actual_result=result, rtol=0.05, atol=0.05)


# ─────────────────────────────────────────────────────────────────────────────
# log, log2, log10, log1p — ULP on their positive/(-1,∞) domain; fast allclose
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op, golden_fn, ulp",
    [
        (ttnn.log, torch.log, 1),
        (ttnn.log2, torch.log2, 1),
        (ttnn.log10, torch.log10, 2),
    ],
    ids=["log", "log2", "log10"],
)
def test_log_family_accurate(device, ttnn_op, golden_fn, ulp):
    """log, log2 and log10 in accurate mode over the positive bfloat16 values.

    | Op        | Input range     | Check   | Reason                         |
    |-----------|-----------------|---------|--------------------------------|
    | log, log2 | [SNB, MAX_BF16] | ULP ≤ 1 |                                |
    | log10     | [SNB, MAX_BF16] | ULP ≤ 2 | extra rounding from the change |
    |           |                 |         | of base multiply               |

    x <= 0 is in test_log_family_nonpositive_domain.
    """
    input_tensor = generate_float32_bits_in_range(SMALLEST_NORMAL_BF16, MAX_BF16)
    tt_in = to_float32_tt_tensor(input_tensor, device)

    golden = golden_fn(input_tensor.float()).to(torch.float32)
    result = ttnn.to_torch(ttnn_op(tt_in, fast_and_approximate_mode=False)).to(torch.float32)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=ulp, allow_nonfinite=True)


@pytest.mark.parametrize("ttnn_op", [ttnn.log, ttnn.log2, ttnn.log10], ids=["log", "log2", "log10"])
@pytest.mark.parametrize("fast", [False, True], ids=["accurate", "fast"])
def test_log_family_nonpositive_domain(device, ttnn_op, fast):
    """log, log2 and log10 outside their domain, both modes.

    | Input range       | Check      | Reason                           |
    |-------------------|------------|----------------------------------|
    | [-MAX_BF16, 0]    | non-finite | log(0) = -inf, log(x < 0) = NaN  |
    """
    input_tensor = generate_float32_bits_in_range(-MAX_BF16, 0.0)
    tt_in = to_float32_tt_tensor(input_tensor, device)

    result = ttnn.to_torch(ttnn_op(tt_in, fast_and_approximate_mode=fast))
    assert_all_nonfinite(result, f"{ttnn_op.__name__}(x<=0)")


def test_log1p_accurate(device):
    """Float32 sweep over every normal bfloat16 value for log1p, accurate mode.

    | Input range | Check      | Reason                                  |
    |-------------|------------|-----------------------------------------|
    | x > -1      | ULP ≤ 1    |                                         |
    | x <= -1     | non-finite | log1p(-1) = -inf, log1p(x < -1) = NaN   |
    """
    input_tensor = generate_float32_bits()  # all normals; specials→0
    tt_in = to_float32_tt_tensor(input_tensor, device)

    golden = torch.log1p(input_tensor.float()).to(torch.float32)
    result = ttnn.to_torch(ttnn.log1p(tt_in, fast_and_approximate_mode=False)).to(torch.float32)

    x = input_tensor.float()
    in_domain = x > -1.0
    assert_with_ulp(
        expected_result=golden[in_domain], actual_result=result[in_domain], ulp_threshold=1, allow_nonfinite=True
    )

    out_of_domain = x <= -1.0
    assert out_of_domain.any(), "expected x <= -1 band to be non-empty for this exhaustive sweep"
    assert_all_nonfinite(result[out_of_domain], "log1p(x<=-1)")


@pytest.mark.parametrize(
    "ttnn_op", [ttnn.log, ttnn.log2, ttnn.log10, ttnn.log1p], ids=["log", "log2", "log10", "log1p"]
)
def test_log_family_fast(device, ttnn_op):
    """log, log2, log10 and log1p in fast mode.

    | Input range | Check                           |
    |-------------|---------------------------------|
    | [1, 100]    | atol 0.0625 (rtol default 1e-5) |
    """
    input_tensor = generate_float32_bits_in_range(1.0, 100.0)
    tt_in = to_float32_tt_tensor(input_tensor, device)

    golden_fn = {ttnn.log: torch.log, ttnn.log2: torch.log2, ttnn.log10: torch.log10, ttnn.log1p: torch.log1p}[ttnn_op]
    golden = golden_fn(input_tensor.float()).to(torch.float32)
    result = ttnn.to_torch(ttnn_op(tt_in, fast_and_approximate_mode=True)).to(torch.float32)

    assert_allclose(expected_result=golden, actual_result=result, atol=0.0625)


# ─────────────────────────────────────────────────────────────────────────────
# mish — x * tanh(softplus(x)); compound, allclose in both modes
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fast", [False, True], ids=["accurate", "fast"])
def test_mish(device, fast):
    """Float32 sweep over every normal bfloat16 value for mish = x * tanh(softplus(x)), both modes.

    | Input range | Check                | Reason                                      |
    |-------------|----------------------|---------------------------------------------|
    | all         | rtol 1e-5, atol 0.02 | looser than the old tests' 0.008 because    |
    |             |                      | the sweep hits the trough near x ≈ -1.19    |
    """
    input_tensor = generate_float32_bits()  # all normals; specials→0
    tt_in = to_float32_tt_tensor(input_tensor, device)

    golden = F.mish(input_tensor.float()).to(torch.float32)
    result = ttnn.to_torch(ttnn.mish(tt_in, fast_and_approximate_mode=fast)).to(torch.float32)

    assert_allclose(expected_result=golden, actual_result=result, rtol=1e-5, atol=0.02)

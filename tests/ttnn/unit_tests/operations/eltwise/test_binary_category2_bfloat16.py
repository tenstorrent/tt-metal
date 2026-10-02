# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_equal, assert_with_ulp
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    pairwise_inputs,
    run_binary,
)

pytestmark = pytest.mark.use_module_device

"""
Category 2: binary comparison + min/max

 1. ttnn.eq               - Equal
 2. ttnn.ne               - Not equal
 3. ttnn.lt               - Less than
 4. ttnn.le               - Less than or equal
 5. ttnn.gt               - Greater than
 6. ttnn.ge               - Greater than or equal
 7. ttnn.isclose          - Close within atol/rtol
 8. ttnn.minimum          - Elementwise min
 9. ttnn.maximum          - Elementwise max

Accuracy criteria
─────────────────
  eq, ne, lt, le, gt, ge : exact  (SFPU comparison; result is 0 or 1)
  isclose                : exact vs torch.isclose after two documented
                           dest-precision exceptions (see test_isclose)
  minimum, maximum       : exact on finite values, ±0, and ±inf
                           (see test_minmax_ops). NaN / signed-zero / inf
                           edge cases live in test_minmax_special_values.
"""


@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.eq,
        ttnn.ne,
        ttnn.lt,
        ttnn.le,
        ttnn.gt,
        ttnn.ge,
        ttnn.eq_,
        ttnn.ne_,
        ttnn.lt_,
        ttnn.le_,
        ttnn.gt_,
        ttnn.ge_,
    ],
)
def test_relational_ops(device, ttnn_op):
    """Pairwise coverage of relational ops over the stratified bfloat16 grid.

    The grid includes ±0, ±inf, and one qNaN. SFPU comparison matches IEEE:
    ±0 compare equal, NaN makes every ordered/eq compare false (ne true),
    and matching-sign infs compare equal.
    Device returns 0/1 in the input dtype; golden is bool.
    """
    input_a, input_b = pairwise_inputs(include_spl_values=True)
    golden, result = run_binary(device, ttnn_op, input_a, input_b)
    assert_equal(golden.float(), result.float())


@pytest.mark.parametrize("ttnn_op", [ttnn.minimum, ttnn.maximum])
def test_minmax_ops(device, ttnn_op):
    """Pairwise coverage of ttnn.minimum / ttnn.maximum over the stratified grid.

    Output is a select of one operand, so finite values, ±0, and ±inf match
    torch at 0 ULP. NaN pairs are excluded here — SFPSWAP does not propagate
    NaN the way torch does; that contract is asserted independently in
    test_minmax_special_values.
    """
    input_a, input_b = pairwise_inputs(include_spl_values=False)
    golden, result = run_binary(device, ttnn_op, input_a, input_b)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=0)


@pytest.mark.parametrize(
    "rtol, atol",
    [
        (1e-05, 1e-08),  # torch / ttnn defaults
        (0.0, 0.0),  # exact equality, plus matching-sign inf
        (1e-04, 0.0),  # relative-only
        (0.0, 1.0),  # absolute-only, large enough to fire on this grid
    ],
)
@pytest.mark.parametrize("equal_nan", [False, True])
def test_isclose(device, rtol, atol, equal_nan):
    """Pairwise coverage of ttnn.isclose over the stratified bfloat16 grid.

    Kernel: |a - b| <= atol + rtol * |b|, with an explicit Inf/NaN fix-up
    matching torch.isclose (equal_nan selects both-NaN => 1).

    Two dest-precision exceptions, both empty on the default (1e-5, 1e-8).
    Expected is built from the inputs and device arithmetic, not from whether
    the observed result already disagrees with torch:

    1) Underflow FTZ of |a-b|: dest packing flushes subnormal differences
       (|a-b|_fp32 < 2^{-126}) to 0, so isclose is True while torch still
       sees a nonzero gap. Min-normal |a-b| = 2^{-126} is representable and
       stays nonzero. Typical on this grid when both |a| and |b| sit in
       [2^{-126}, 2^{-119}).
       Example (atol=0): 1.175e-38 vs 1.185e-38 → torch False, device True.

    2) Tolerance fence (atol > 0 only): SFPU |a-b| is fp32 dest; torch.isclose
       on bf16 rounds |a-b| to bf16. When the rounded difference equals atol
       but the fp32 difference is slightly larger, torch says close and the
       device does not. At atol=0 this predicate is dest FTZ, not a fence.
       Example (atol=1): 6.007e-08 vs -1.0 → torch |a-b|_bf16 = 1.0,
       device |a-b|_fp32 = 1.00000012.
    """
    input_a, input_b = pairwise_inputs(include_spl_values=True)
    golden, result = run_binary(
        device, ttnn.isclose, input_a, input_b, golden_kwargs={"rtol": rtol, "atol": atol, "equal_nan": equal_nan}
    )

    golden = golden.float()
    result = result.float()

    # Dest FTZ: subnormal |a-b| in dest packs to 0, so isclose is True.
    # Min-normal |a-b| = 2^{-126} is kept, so the bound is strict.
    diff_fp32 = (input_a.float() - input_b.float()).abs()
    dest_sub_flushed = torch.isfinite(input_a) & torch.isfinite(input_b) & (diff_fp32 < (2.0**-126))
    adjusted_golden = torch.where(dest_sub_flushed, torch.ones_like(golden), golden)

    # Tolerance fence (atol > 0 only): torch |a-b| rounded to bf16 equals
    # atol, fp32 dest difference is strictly larger. At atol=0 this is the
    # FTZ case above (diff_bf16 == 0 and diff_fp32 > 0), not a fence.
    diff_bf16 = (input_a - input_b).abs()
    atol_bf16 = torch.tensor(atol, dtype=torch.bfloat16)
    fence = (atol > 0) & (diff_bf16 == atol_bf16) & (diff_fp32 > float(atol))
    adjusted_golden = torch.where(fence, torch.zeros_like(adjusted_golden), adjusted_golden)

    assert_equal(adjusted_golden, result)


@pytest.mark.parametrize("ttnn_op, is_max", [(ttnn.minimum, False), (ttnn.maximum, True)])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_minmax_special_values(device, ttnn_op, is_max, dtype):
    """Cross product of ±0, ±1, ±inf, NaN for ttnn.minimum / ttnn.maximum.

    bfloat16 dest uses SFPSWAP (sign-magnitude, NaN as +inf). Operand order
    does not change this: minimum(nan, x) matches minimum(x, nan). Dest packing
    also drops the zero sign bit.

    call                                  torch    ttnn bf16    ttnn fp32
    ------------------------------------  -------  -----------  -----------
    minimum(x, nan)   x finite or -inf    NaN      x            x
    minimum(+inf, nan)                    NaN      +inf         +inf
    minimum(nan, nan)                     NaN      +inf         NaN
    maximum(x, nan)   any x               NaN      +inf         NaN

    float32 maximum matches torch (IEEE NaN-propagate). float32 minimum still
    treats one-sided NaN as +inf; both-NaN stays NaN. IEEE min(+0, -0) is -0
    and max(-0, -0) is -0; bf16 dest writes +0, fp32 dest keeps a sign bit.
    """
    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)

    special_values = [0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan")]
    x_vals = [x for x in special_values for _ in special_values]
    y_vals = [y for _ in special_values for y in special_values]

    input_a = torch.tensor(x_vals, dtype=torch_dtype)
    input_b = torch.tensor(y_vals, dtype=torch_dtype)

    inf = torch.tensor(float("inf"), dtype=torch_dtype)
    a_cmp = torch.where(torch.isnan(input_a), inf, input_a)
    b_cmp = torch.where(torch.isnan(input_b), inf, input_b)

    if dtype == "float32" and is_max:
        expected = torch.maximum(input_a, input_b)
    else:
        expected = torch.maximum(a_cmp, b_cmp) if is_max else torch.minimum(a_cmp, b_cmp)
        if dtype == "float32":
            expected = torch.where(torch.isnan(input_a) & torch.isnan(input_b), input_a, expected)
        else:
            expected = torch.where(expected == 0, torch.zeros_like(expected), expected)

    tt_a = ttnn.from_torch(input_a, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(input_b, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn_op(tt_a, tt_b))

    assert_with_ulp(expected_result=expected, actual_result=result, ulp_threshold=0, allow_nonfinite=True)
    if dtype == "bfloat16":
        zero = result == 0
        assert not torch.signbit(result[zero]).any()


@pytest.mark.parametrize(
    "op_name",
    [
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        "float32",
        "bfloat16",
    ],
)
def test_special_values(device, op_name, dtype):
    """
    Comprehensive test for special floating-point values: 0, -0, inf, -inf, nan
    Tests all combinations of these values as inputs to a binary operation.
    """
    torch_fn = getattr(torch, op_name)
    ttnn_fn = getattr(ttnn, op_name)

    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)

    # Special values to test
    special_values = [0.0, float("inf"), float("-inf"), float("nan"), 1.0, -1.0, -0.0]

    # Create all combinations
    x_vals = [x for x in special_values for _ in special_values]
    y_vals = [y for _ in special_values for y in special_values]

    x_torch = torch.tensor(x_vals, dtype=torch_dtype)
    y_torch = torch.tensor(y_vals, dtype=torch_dtype)
    golden = torch_fn(x_torch, y_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn_fn(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt)

    assert_equal(golden.float(), tt_out.float())

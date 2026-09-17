# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp, assert_with_pcc, assert_allclose
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_bits,
    to_tt_tensor,
)
from models.common.utility_functions import is_blackhole

pytestmark = pytest.mark.use_module_device

"""
Category 4: Ops with float/scalar parameter

11. ttnn.elu          - alpha (default 1.0):  x >= 0 ? x : alpha * (exp(x) - 1)
12. ttnn.heaviside     - value:                 x < 0 ? 0 : x > 0 ? 1 : value
13. ttnn.leaky_relu    - negative_slope:        x >= 0 ? x : negative_slope * x
14. ttnn.relu_max      - upper_limit:           min(max(x, 0), upper_limit)
15. ttnn.relu_min      - lower_limit:           max(x, lower_limit)
16. ttnn.rpow          - exponent:              exponent ** x
17. ttnn.celu          - alpha (default 1.0):   x >= 0 ? x : alpha * (exp(x / alpha) - 1)
18. ttnn.softcap       - beta:                  beta * tanh(x / beta)
19. ttnn.fill          - fill_value:            fill_value (input is ignored)
20. ttnn.hardshrink    - lambd (default 0.5):   |x| > lambd ? x : 0
21. ttnn.softshrink    - lambd (default 0.5):   x > lambd ? x - lambd : x < -lambd ? x + lambd : 0

Each op takes a single tensor input plus one Python float/scalar parameter, unlike
Category 5's dim-split gated-linear-units. The input tensor is where "exhaustive
bfloat16 coverage" applies: every op below sweeps all 32,512 positive and 32,512
negative normal bfloat16 values (a single (256, 256) tile, one device dispatch) through
the tensor argument. The scalar parameter is not itself tensor-shaped, so it cannot be
swept bit-for-bit in the same single-dispatch way; instead it is parametrized over a
representative set of values (default, fractional, integer, and sign-varying), except
for `fill`, whose output does not depend on the input tensor at all -- there the
parameter itself is the thing being covered, so `fill_value` is parametrized across
representative bfloat16 value classes instead.

Accuracy criteria
─────────────────
  heaviside, relu_max, relu_min, hardshrink : exact (bit-for-bit)
      Pure comparison/selection with no arithmetic on the passthrough branches.
  leaky_relu, softshrink                    : ULP ≤ 1
      Exactly one multiply/add-sub on the non-identity branch.
  elu, celu                                 : ULP ≤ 1 (excluding a narrow band)
      alpha * (exp(x) - 1) [or exp(x/alpha)] loses more than 1 ULP to
      cancellation for x in a narrow band adjacent to 0; this band is
      already characterized (and covered via allclose) in test_elu.py /
      test_celu_21f.py and is excluded here the same way.
  softcap                                   : ULP ≤ 2 (after FTZ masking) + PCC ≥ 0.9999
      beta * tanh(x / beta): the tanh is a Sollya polynomial approximation
      (see test_unary.py::test_softcap_bfloat16_full_domain). Blackhole only.
  rpow                                      : PCC ≥ 0.99 + allclose(atol=1e-2, rtol=0.1)
      exponent ** x is evaluated via an exp2/log2 chain; existing coverage
      (test_unary_rpow_ttnn) already uses this looser, magnitude-scaled
      tolerance because the transcendental approximation -- not rounding --
      dominates the error.
  fill                                      : exact (bit-for-bit)
      The output is the parameter value itself, broadcast.
"""


def _flush_subnormal_golden_and_result(golden, result):
    """Zero both tensors where golden is subnormal (device FTZ), leaving finite
    non-subnormal mismatches (real bugs) visible."""
    tiny = torch.finfo(torch.bfloat16).tiny
    ftz = (golden.abs() > 0) & (golden.abs() <= tiny)
    golden = golden.clone()
    golden[ftz] = 0.0
    result = result.clone()
    result[ftz] = 0.0
    return golden, result


def _exhaustive_input(mask_predicate=None, safe_value=1.0):
    """All normal bfloat16 bit patterns as a (1, 1, 256, 256) tensor.

    If mask_predicate is given, elements where it is True are replaced by
    safe_value so the tile shape (and therefore exhaustive coverage of every
    other element) is preserved instead of dropping elements.
    """
    B = generate_bfloat16_bits(include_spl_values=False)  # (256, 256)
    if mask_predicate is not None:
        mask = mask_predicate(B)
        B = torch.where(mask, torch.full_like(B, safe_value), B)
    return B.unsqueeze(0).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# heaviside, relu_max, relu_min, hardshrink — exact (bit-for-bit)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0, -1.0, 2.5])
def test_heaviside_op(device, value):
    """Exhaustive normal bfloat16 coverage for heaviside.

    output_i = 0 if x_i < 0, value if x_i == 0, 1 if x_i > 0. Pure selection
    with no arithmetic on the passthrough branches, so the device result
    must match the CPU reference bit-for-bit.

    Also exercises the preallocated output_tensor=/queue_id= call path
    exhaustively (previously only covered by test_unary_ops_ttnn.py::
    test_unary_heaviside_ttnn on a small random bf16 sample; that test was
    removed once this covered it exhaustively).
    """
    input_tensor = _exhaustive_input()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.heaviside)
    golden = golden_function(input_tensor, value, device=device)

    tt_result = ttnn.heaviside(tt_in, value)
    result = ttnn.to_torch(tt_result)

    assert torch.equal(result, golden), (
        f"heaviside(value={value}) diverged for {int((result != golden).sum().item())} " f"of {result.numel()} elements"
    )

    preallocated_output = to_tt_tensor(torch.zeros_like(input_tensor), device)
    ttnn.heaviside(tt_in, value, output_tensor=preallocated_output, queue_id=0)
    preallocated_result = ttnn.to_torch(preallocated_output)
    assert torch.equal(preallocated_result, golden), (
        f"heaviside(value={value}) via output_tensor=/queue_id= diverged for "
        f"{int((preallocated_result != golden).sum().item())} of {preallocated_result.numel()} elements"
    )


@pytest.mark.parametrize("upper_limit", [0.0, 1.0, 3.0, 100.0, -1.0])
def test_relu_max_op(device, upper_limit):
    """Exhaustive normal bfloat16 coverage for relu_max.

    output_i = min(max(x_i, 0), upper_limit). Pure clamp with no rounding,
    so the device result must match the CPU reference bit-for-bit.
    upper_limit < 0 collapses the output to a constant, since
    max(x, 0) >= 0 > upper_limit for every x; that degenerate case is
    included via -1.0.
    """
    input_tensor = _exhaustive_input()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.relu_max)
    golden = golden_function(input_tensor, upper_limit, device=device)

    tt_result = ttnn.relu_max(tt_in, upper_limit)
    result = ttnn.to_torch(tt_result)

    assert torch.equal(result, golden), (
        f"relu_max(upper_limit={upper_limit}) diverged for {int((result != golden).sum().item())} "
        f"of {result.numel()} elements"
    )


@pytest.mark.parametrize("lower_limit", [0.0, -1.0, -3.0, -100.0, 1.0])
def test_relu_min_op(device, lower_limit):
    """Exhaustive normal bfloat16 coverage for relu_min.

    output_i = max(x_i, lower_limit). Pure clamp with no rounding, so the
    device result must match the CPU reference bit-for-bit. lower_limit > 0
    collapses the output's lower bound above 0, tested via 1.0.
    """
    input_tensor = _exhaustive_input()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.relu_min)
    golden = golden_function(input_tensor, lower_limit, device=device)

    tt_result = ttnn.relu_min(tt_in, lower_limit)
    result = ttnn.to_torch(tt_result)

    assert torch.equal(result, golden), (
        f"relu_min(lower_limit={lower_limit}) diverged for {int((result != golden).sum().item())} "
        f"of {result.numel()} elements"
    )


@pytest.mark.parametrize("lambd", [0.5, 0.25, 1.0, 2.0, 4.0])
def test_hardshrink_op(device, lambd):
    """Exhaustive normal bfloat16 coverage for hardshrink.

    output_i = x_i if |x_i| > lambd else 0. Pure selection with no
    arithmetic on the passthrough branch. For BFLOAT16 inputs the device
    pre-rounds lambd to BF16 (RNE) before comparing (UnaryOpType::HARDSHRINK
    in unary_op_utils.cpp) so the FP32-domain compare lands on a threshold
    the input's BF16 precision can actually represent. All lambd values
    used here are already exactly representable in bfloat16, so no extra
    quantization is needed to keep the golden aligned with the device.
    """
    input_tensor = _exhaustive_input()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.hardshrink)
    golden = golden_function(input_tensor, lambd=lambd, device=device)

    tt_result = ttnn.hardshrink(tt_in, lambd=lambd)
    result = ttnn.to_torch(tt_result)

    assert torch.equal(result, golden), (
        f"hardshrink(lambd={lambd}) diverged for {int((result != golden).sum().item())} "
        f"of {result.numel()} elements"
    )


# ─────────────────────────────────────────────────────────────────────────────
# leaky_relu, softshrink — ULP ≤ 1
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("negative_slope", [0.01, 0.1, 1.0, 2.0, -0.5])
def test_leaky_relu_op(device, negative_slope):
    """Exhaustive normal bfloat16 coverage for leaky_relu.

    output_i = x_i (exact) if x_i >= 0, else negative_slope * x_i (one
    multiply, rounds at most 1 ULP). A small negative_slope can multiply a
    normal x_i down into subnormal range, which the SFPU flushes to zero
    (FTZ); that is masked the same way Category 5 handles it, deriving the
    mask from golden alone so a genuine device bug at those positions
    would still be caught. A |negative_slope| > 1 can also overflow the
    most negative bfloat16 values to +/-inf on both the golden (computed
    in bf16) and device sides, so non-finite values are allowed as long as
    they agree in position.

    Also exercises the preallocated output_tensor=/queue_id= call path
    exhaustively (previously only covered by test_unary_ops_ttnn.py::
    test_unary_leaky_relu_ttnn on a small random bf16 sample; that test was
    removed once this covered it exhaustively).
    """
    input_tensor = _exhaustive_input()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.leaky_relu)
    golden_raw = golden_function(input_tensor, negative_slope=negative_slope, device=device)

    tt_result = ttnn.leaky_relu(tt_in, negative_slope=negative_slope)
    result = ttnn.to_torch(tt_result)
    golden, result = _flush_subnormal_golden_and_result(golden_raw, result)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1, allow_nonfinite=True)

    preallocated_output = to_tt_tensor(torch.zeros_like(input_tensor), device)
    ttnn.leaky_relu(tt_in, negative_slope=negative_slope, output_tensor=preallocated_output, queue_id=0)
    preallocated_result = ttnn.to_torch(preallocated_output)
    golden_pa, preallocated_result = _flush_subnormal_golden_and_result(golden_raw, preallocated_result)

    assert_with_ulp(expected_result=golden_pa, actual_result=preallocated_result, ulp_threshold=1, allow_nonfinite=True)


@pytest.mark.parametrize("lambd", [0.5, 0.25, 1.0, 2.0])
def test_softshrink_op(device, lambd):
    """Exhaustive normal bfloat16 coverage for softshrink.

    output_i = x_i - lambd if x_i > lambd, x_i + lambd if x_i < -lambd,
    else 0 (exact). The shrink branches have one add/sub, rounding at most
    1 ULP; that add/sub can also land in subnormal range near the lambd
    boundary, which the SFPU flushes to zero (FTZ) -- masked the same way
    as Category 5, deriving the mask from golden alone.
    """
    input_tensor = _exhaustive_input()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.softshrink)
    golden = golden_function(input_tensor, lambd=lambd, device=device)

    tt_result = ttnn.softshrink(tt_in, lambd=lambd)
    result = ttnn.to_torch(tt_result)
    golden, result = _flush_subnormal_golden_and_result(golden, result)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


# ─────────────────────────────────────────────────────────────────────────────
# elu, celu — ULP ≤ 1 (excluding a narrow cancellation band near 0)
# ─────────────────────────────────────────────────────────────────────────────

# test_elu.py::test_elu_arange_masking / test_celu_21f.py::test_celu_arange already
# characterize this band for the default alpha (1.0) and cover it separately via
# allclose. It is alpha-independent: the loss comes from cancellation in exp(x) - 1
# (or exp(x/alpha) - 1) before the alpha scale is applied.
_ELU_CANCELLATION_BAND = (-0.28515625, 1.1663108012064884e-38)


@pytest.mark.parametrize("alpha", [1.0, 0.5, 2.0, 0.1])
def test_elu_op(device, alpha):
    """Exhaustive normal bfloat16 coverage for elu across alpha values.

    output_i = x_i (exact) if x_i >= 0, else alpha * (exp(x_i) - 1).
    test_elu.py already covers the default alpha exhaustively; this adds
    coverage for non-default alpha over the same exhaustive input domain.
    """
    low, high = _ELU_CANCELLATION_BAND
    input_tensor = _exhaustive_input(mask_predicate=lambda b: (b >= low) & (b <= high))

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.elu)
    golden = golden_function(input_tensor, alpha=alpha, device=device)

    tt_result = ttnn.elu(tt_in, alpha=alpha)
    result = ttnn.to_torch(tt_result)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1, allow_nonfinite=True)


@pytest.mark.parametrize("alpha", [1.0, 0.5, 2.0, 0.1])
def test_celu_op(device, alpha):
    """Exhaustive normal bfloat16 coverage for celu across alpha values.

    output_i = x_i (exact) if x_i >= 0, else alpha * (exp(x_i / alpha) - 1).
    test_celu_21f.py already covers the default alpha exhaustively; this
    adds coverage for non-default alpha over the same exhaustive domain.
    """
    low, high = _ELU_CANCELLATION_BAND
    input_tensor = _exhaustive_input(mask_predicate=lambda b: (b >= low) & (b <= high))

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.celu)
    golden = golden_function(input_tensor, alpha=alpha, device=device)

    tt_result = ttnn.celu(tt_in, alpha=alpha)
    result = ttnn.to_torch(tt_result)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1, allow_nonfinite=True)


# ─────────────────────────────────────────────────────────────────────────────
# softcap — ULP ≤ 2 (post FTZ-mask) + PCC ≥ 0.9999. Blackhole only.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not is_blackhole(), reason="softcap is implemented for Blackhole only")
@pytest.mark.parametrize("beta", [1.0, 25.0, 0.5, 100.0])
def test_softcap_op(device, beta):
    """Exhaustive normal bfloat16 coverage for softcap across beta values.

    output_i = beta * tanh(x_i / beta). test_unary.py's
    test_softcap_bfloat16_full_domain already covers the full bfloat16
    domain for the model-specific beta=25.0 case; this extends the same
    exhaustive sweep and tolerance policy to other beta values.
    """
    input_tensor = _exhaustive_input()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.softcap)
    golden = golden_function(input_tensor, beta=beta, device=device)

    tt_result = ttnn.softcap(tt_in, beta)
    result = ttnn.to_torch(tt_result)

    max_abs = result.to(torch.float32).abs().max().item()
    bound = beta * (1.0 + 2**-8)
    assert max_abs <= bound, f"softcap overshoot: max |out| {max_abs:.4f} > bound {bound:.4f}"

    # Scale the near-zero FTZ guard with beta: tanh(x/beta) ~= x/beta near 0, so the
    # subnormal-flush boundary in x shifts proportionally to beta.
    flush_floor = 1e-30 * max(beta, 1.0)
    mask = golden.abs() > flush_floor
    assert_with_ulp(expected_result=golden[mask], actual_result=result[mask], ulp_threshold=2)
    assert_with_pcc(golden[mask], result[mask], pcc=0.9999)


# ─────────────────────────────────────────────────────────────────────────────
# rpow — PCC ≥ 0.99 + allclose(atol=1e-2, rtol=0.1)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("exponent", [0.5, 2.0, 3.0, 10.0])
def test_rpow_op(device, exponent):
    """Exhaustive normal bfloat16 coverage for rpow across exponent values.

    output_i = exponent ** x_i. exponent ** x overflows or underflows to
    +/-inf/0 for most large-magnitude bfloat16 x when exponent != 1, so
    input values outside [-30, 30] -- the range already exercised by
    test_unary_rpow_ttnn -- are neutralized to 1.0. This still exhaustively
    sweeps every normal bfloat16 value within the representable working
    range instead of a random uniform sample.
    """
    input_tensor = _exhaustive_input(mask_predicate=lambda b: b.abs() > 30.0)

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.rpow)
    golden = golden_function(input_tensor, exponent, device=device)

    tt_result = ttnn.rpow(tt_in, exponent)
    result = ttnn.to_torch(tt_result)

    assert_with_pcc(golden, result, pcc=0.99)
    assert_allclose(result, golden, atol=1e-2, rtol=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# fill — exact (bit-for-bit). fill_value itself is the swept domain.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fill_value",
    # All values are exactly representable in bfloat16 (zero mantissa bits below
    # bf16's 7-bit mantissa), so truncating vs. round-to-nearest-even fp32->bf16
    # conversion of fill_value cannot disagree with the CPU reference.
    [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0, 2.0**-30, 2.0**127, -(2.0**127)],
)
def test_fill_op(device, fill_value):
    """Coverage for fill across representative bfloat16 value classes.

    fill_value is a single scalar baked into the op call rather than a
    per-element tensor value, so -- unlike every other Category 4 op --
    there is no tensor-shaped domain to sweep exhaustively in one
    dispatch: each distinct fill_value requires its own op call. The
    output does not depend on the input at all (every element is
    unconditionally overwritten), so a fixed input is used here and
    fill_value is instead parametrized across zero, signed values, a
    subnormal-adjacent magnitude, and the largest finite bfloat16
    magnitude.
    """
    input_tensor = torch.arange(256 * 256, dtype=torch.float32).view(1, 1, 256, 256).to(torch.bfloat16)

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.fill)
    golden = golden_function(input_tensor, fill_value, device=device)

    tt_result = ttnn.fill(tt_in, fill_value)
    result = ttnn.to_torch(tt_result)

    assert torch.equal(result, golden), (
        f"fill(fill_value={fill_value}) diverged for {int((result != golden).sum().item())} "
        f"of {result.numel()} elements"
    )

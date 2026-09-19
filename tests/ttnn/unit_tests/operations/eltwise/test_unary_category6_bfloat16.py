# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp, assert_with_pcc
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_bits,
    generate_bfloat16_bits_in_range,
    to_tt_tensor,
    bf16_bits_to_float,
    float_to_bf16_bits,
    SMALLEST_NORMAL_BF16,
    MAX_BF16,
)

pytestmark = pytest.mark.use_module_device

"""
Category 6: Ops with multiple/special parameters

89. ttnn.softplus         - beta (default 1.0), threshold (default 20.0)
90. ttnn.xielu            - alpha_p (default 0.8), alpha_n (default 0.8)
91. ttnn.tanh             - fast_and_approximate_mode (default False)
92. ttnn.sigmoid_accurate - fast_and_approximate_mode (default False) [deprecated]
93. ttnn.sigmoid          - vector_mode (default 4), mode (SigmoidMode enum, default Accurate)
94. ttnn.unary_chain      - ops_chain (list of UnaryWithParam)
95. ttnn.clamp            - min, max (float/int scalar, or Tensor)
96. ttnn.clip             - min, max (float scalar, or Tensor)
97. ttnn.selu             - scale (default 1.0507), alpha (default 1.67326)
98. ttnn.hardtanh         - min_val (default -1.0), max_val (default 1.0)
99. ttnn.threshold        - threshold, value
100. ttnn.tril            - diagonal (default 0)
101. ttnn.triu            - diagonal (default 0)
102. ttnn.round           - decimals (default 0, supported range -6..7)
103. ttnn.polygamma       - k (int, supported range 1..10)
104. ttnn.logit           - eps (optional; eps > 0.5 takes a manual-clamp golden branch)
105. ttnn.rdiv            - value, rounding_mode (None | "floor" | "trunc")
106. ttnn.bitcast         - dtype (bfloat16 <-> uint16 is the only same-bit-width pair for this dtype)

Accuracy criteria
─────────────────
  Exact (bit-for-bit): hardtanh, clamp, clip, tril, triu, bitcast
      Pure comparison/select or bit-reinterpretation, no arithmetic rounding.
      hardtanh's golden accounts for the device's fp32->bf16 truncation of
      min_val/max_val (see the real-hardware quirks note below).
  ULP-gated: threshold (<=1, per the op's own golden comparison config),
      round (<=1), selu (<=1, excluding the elu-style cancellation band near 0
      and the FTZ boundary), rdiv None mode (<=3, excluding the large-|x|
      reciprocal-FTZ band), sigmoid Accurate mode (<=2, excluding the FTZ
      boundary), unary_chain([SQUARE, SQRT]) (<=2, excluding both the
      overflow and underflow bands).
  PCC-gated (transcendental / compound / approximate): tanh, sigmoid_accurate,
      xielu, softplus, logit, polygamma, rdiv floor/trunc modes, sigmoid
      AccurateWithFastExp/FastApproximate modes -- each swept in-domain over
      the exhaustive bf16 grid with out-of-domain or known-unsafe inputs
      excluded via a domain mask (same approach as category3_ulp.py's
      log-family handling). rdiv floor/trunc and the two approximate sigmoid
      modes are PCC- rather than ULP-gated because a single bf16-ULP
      perturbation can flip an integer-valued (floor/trunc) or
      by-design-approximate result by more than a tight ULP bound should
      tolerate.
  Structural (own dedicated tests instead of a single accuracy gate):
      unary_chain (composition of goldens; RELU+TYPECAST is bit-exact outside
      a narrow near-subnormal band, see below).

Notes on golden-function quirks discovered while authoring this file (see
ttnn/ttnn/operations/unary.py):
  - selu's attached golden (_golden_function_selu) ignores the scale/alpha
    kwargs entirely and always calls torch.nn.functional.selu(x) (which bakes
    in the paper defaults 1.0507/1.67326). Since the device kernel *does*
    accept scale/alpha (unary_nanobind.cpp defaults 1.0507f/1.67326f), this
    file computes its own scale/alpha-aware reference instead of relying on
    ttnn.get_golden_function(ttnn.selu) so non-default scale/alpha are
    actually exercised.
  - bitcast only supports same-bit-width dtype pairs; for a bfloat16 input the
    only valid target is uint16 (see the C++ binding's docstring).

Notes on real-hardware quirks discovered by running this file on a device
(all verified against a live wormhole_b0 chip):
  - hardtanh/clamp/clip/tril/triu take their scalar bounds through the same
    fp32->bf16 *truncation* path as unary `fill` (see category4), not
    round-to-nearest-even. hardtanh's min_val/max_val golden is therefore
    built from the truncated bf16 value (bf16_bits_to_float(float_to_bf16_bits(...)))
    instead of a plain python float, matching what the device actually clamps
    against. clamp/clip/tril/triu were only exercised with bounds that are
    already exact in bf16, so they don't need this treatment.
  - ttnn.sigmoid's vector_mode is only valid as C (2) or RC (4) -- R/vector_mode=1
    raises a TT_FATAL. Further, C (2) is a *sub-tile* optimization (process
    only some columns of the tile) intended for narrow, non-tile-full inputs
    (see test_sigmoid_vector_modes.py's narrow out_channels shape); handing it
    a fully-populated 32x32 tile leaves the unprocessed columns holding stale
    DEST content, not a valid sigmoid output. The exhaustive full-tile sweep
    in this file therefore only uses the default vector_mode=4 (RC).
  - ttnn.SigmoidMode.AccurateWithFastExp's fast exp approximation overflows
    for x >= 174 (sigmoid should saturate to 1.0 there); this region is
    swept but excluded from the accuracy assertion, and the mode is PCC-gated
    (like FastApproximate) rather than tight-ULP-gated since it trades
    accuracy for speed by design.
  - rdiv (rounding_mode=None) is implemented as value * reciprocal(x); once
    |x| >= 1/tiny (~8.5e37) the reciprocal(x) sub-computation itself
    underflows to subnormal and is flushed to zero *before* the multiply,
    zeroing the whole product regardless of value's magnitude. This is
    excluded via a domain mask on |x|, mirroring softsign's reciprocal-FTZ
    note in category4. rdiv floor/trunc modes are PCC-gated (not ULP-gated):
    a 1-bf16-ULP difference in the underlying division can flip the
    floor/trunc result by a whole integer near an integer boundary, which is
    not something a relative-ULP metric can sensibly tolerate (matches
    test_unary_rdiv_ttnn's existing convention in test_unary.py).
  - selu/rdiv/sigmoid/unary_chain(SQUARE,SQRT) all hit the same underlying
    "flush-to-zero near the smallest normal bf16" behavior: whichever side
    (golden or device result) lands at-or-below `tiny` is zeroed on both
    sides before comparing, since the hardware and a float64 reference can
    legitimately round to opposite sides of that boundary.
  - unary_chain([SQUARE, SQRT]) underflows for any |x| < sqrt(tiny) (~3.4e-20)
    since x^2 itself underflows to subnormal/zero before the sqrt ever runs;
    this is excluded via a two-sided |x| domain mask (unlike a plain |x|
    computed in float64, which never underflows).
  - unary_chain([RELU, TYPECAST(bfloat16->float32)]) is not bit-exact for the
    lowest ~11 bf16 exponents just above the subnormal boundary (|x| below
    ~2^-115): the widened fp32 result differs from a simple zero-extension of
    the bf16 bit pattern by a small relative factor. This narrow band is
    excluded from the bit-exact comparison.
"""


MAX_BF16_VAL = torch.finfo(torch.bfloat16).max


def _exhaustive_bf16_4d():
    """All 65,536 bfloat16 bit-patterns (finite normals only), shape (1, 1, 256, 256)."""
    return generate_bfloat16_bits(include_spl_values=False).unsqueeze(0).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# hardtanh, clamp, clip, tril, triu, bitcast — exact (bit-for-bit)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("min_val", [0.25, 0.5, 0.66, -1.0])
@pytest.mark.parametrize("max_val", [1.0, 2.5, 3.0, 6.6])
def test_hardtanh_op(device, min_val, max_val):
    """output_i = clamp(x_i, min_val, max_val). Exhaustive normal bf16 sweep.

    The device truncates min_val/max_val to bf16 (same fp32->bf16 truncation
    path as `fill`, not round-to-nearest-even), so the golden is built from
    the truncated bf16 bounds rather than the raw python floats -- otherwise
    e.g. min_val=0.66 rounds to 0.66015625 (RNE) on the golden side but the
    device clamps against 0.65625 (truncated), diverging on every clamped
    element.
    """
    input_tensor = _exhaustive_bf16_4d()
    device_min_val = bf16_bits_to_float(float_to_bf16_bits(min_val))
    device_max_val = bf16_bits_to_float(float_to_bf16_bits(max_val))

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.hardtanh)
    golden = golden_function(input_tensor, min_val=device_min_val, max_val=device_max_val, device=device)

    tt_result = ttnn.hardtanh(tt_in, min_val=min_val, max_val=max_val)
    result = ttnn.to_torch(tt_result)

    assert torch.equal(result, golden), (
        f"hardtanh(min_val={min_val}, max_val={max_val}) diverged for "
        f"{int((result != golden).sum().item())} of {result.numel()} elements"
    )


@pytest.mark.parametrize(
    "min_val, max_val",
    [
        (0.0, 1.0),
        (None, 1.0),
        (-1.0, None),
        (-5, 5),  # ints
        (2.0, 1.0),  # degenerate: min > max collapses to a constant
    ],
    ids=["both", "min_only", "max_only", "int_bounds", "degenerate"],
)
@pytest.mark.parametrize("ttnn_op", [ttnn.clamp, ttnn.clip])
def test_clamp_clip_scalar_ops(device, ttnn_op, min_val, max_val):
    """output_i = clamp(x_i, min_val, max_val) with scalar float/int bounds."""
    input_tensor = _exhaustive_bf16_4d()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, min_val, max_val, device=device)

    tt_result = ttnn_op(tt_in, min_val, max_val)
    result = ttnn.to_torch(tt_result)

    assert torch.equal(result, golden), (
        f"{ttnn_op.__name__}(min={min_val}, max={max_val}) diverged for "
        f"{int((result != golden).sum().item())} of {result.numel()} elements"
    )


@pytest.mark.parametrize("ttnn_op", [ttnn.clamp, ttnn.clip])
def test_clamp_clip_tensor_bounds(device, ttnn_op):
    """output_i = clamp(x_i, min_i, max_i) with per-element Tensor bounds.

    min/max are built from the same exhaustive bf16 grid (transposed / negated)
    so bounds vary per-element instead of being a single scalar.
    """
    B = generate_bfloat16_bits(include_spl_values=False)  # (256, 256)
    input_tensor = B.unsqueeze(0).unsqueeze(0)
    min_tensor = (-B.abs()).unsqueeze(0).unsqueeze(0)
    max_tensor = (B.abs()).unsqueeze(0).unsqueeze(0)

    tt_in = to_tt_tensor(input_tensor, device)
    tt_min = to_tt_tensor(min_tensor, device)
    tt_max = to_tt_tensor(max_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, min_tensor, max_tensor, device=device)

    tt_result = ttnn_op(tt_in, tt_min, tt_max)
    result = ttnn.to_torch(tt_result)

    assert torch.equal(result, golden), (
        f"{ttnn_op.__name__}(Tensor min/max) diverged for "
        f"{int((result != golden).sum().item())} of {result.numel()} elements"
    )


def _tril_triu_input():
    """A (1, 1, 96, 96) tensor (3x3 tiles) filled with the exhaustive bf16 grid,
    tiled to cover every position with a repeating, but full-range, pattern."""
    B = generate_bfloat16_bits(include_spl_values=False)  # (256, 256) -- 65,536 unique values
    flat = B.flatten()[: 96 * 96]
    return flat.view(1, 1, 96, 96)


@pytest.mark.parametrize("diagonal", [-100, -50, -10, -1, 0, 1, 10, 50, 100])
@pytest.mark.parametrize("ttnn_op", [ttnn.tril, ttnn.triu])
def test_tril_triu_ops(device, ttnn_op, diagonal):
    """Row/column-index-dependent masking, not a per-value sweep. diagonal spans
    from below the matrix (zeros out most/all for tril's complement side) to
    above it (keeps everything), including exactly at the matrix bounds (+-96).

    NOTE: the generic golden wrapper (_golden_function in unary.py) forwards
    only *args to the underlying torch function, silently dropping any
    keyword arguments -- so `diagonal` must be passed positionally here, or
    the golden always falls back to torch.tril/triu's default diagonal=0.
    """
    input_tensor = _tril_triu_input()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, diagonal, device=device)

    tt_result = ttnn_op(tt_in, diagonal=diagonal)
    result = ttnn.to_torch(tt_result)

    assert torch.equal(result, golden), (
        f"{ttnn_op.__name__}(diagonal={diagonal}) diverged for "
        f"{int((result != golden).sum().item())} of {result.numel()} elements"
    )


def test_bitcast_op(device):
    """bitcast reinterprets bits without conversion. For a bfloat16 input, the
    only same-bit-width target dtype is uint16 (per the C++ binding's doc)."""
    input_tensor = _exhaustive_bf16_4d()

    tt_in = to_tt_tensor(input_tensor, device)
    golden = (
        input_tensor.view(torch.int16).view(torch.uint16)
        if hasattr(torch, "uint16")
        else input_tensor.view(torch.int16)
    )

    tt_result = ttnn.bitcast(tt_in, ttnn.uint16)
    result = ttnn.to_torch(tt_result)

    # Compare via the raw 16-bit pattern (torch has limited uint16 support on
    # some builds, so bounce through int16 for a dtype-agnostic bit compare).
    golden_bits = input_tensor.view(torch.int16)
    result_bits = result.to(torch.int16) if result.dtype != torch.int16 else result

    assert torch.equal(result_bits, golden_bits), (
        f"bitcast(bfloat16 -> uint16) diverged for "
        f"{int((result_bits != golden_bits).sum().item())} of {result_bits.numel()} elements"
    )


# ─────────────────────────────────────────────────────────────────────────────
# threshold, round, selu, rdiv — ULP-gated
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "threshold_val, value",
    [(0.0, 0.0), (0.5, -1.0), (-1.0, 0.0), (3.0, 100.0), (-100.0, -100.0)],
)
def test_threshold_op(device, threshold_val, value):
    """output_i = x_i if x_i > threshold_val else value. Golden pins ULP<=1
    (degenerate scope) itself via set_golden_comparison_config."""
    input_tensor = _exhaustive_bf16_4d()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.threshold)
    golden = golden_function(input_tensor, threshold_val, value, device=device)

    tt_result = ttnn.threshold(tt_in, threshold_val, value)
    result = ttnn.to_torch(tt_result)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


@pytest.mark.parametrize("decimals", [None, 0, -1, -3, 6])
def test_round_op(device, decimals):
    """Exhaustive normal bf16 sweep. decimals in [-6, 7] is the supported
    range; representative values from both signs are used here."""
    input_tensor = _exhaustive_bf16_4d()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.round)
    if decimals is None:
        golden = golden_function(input_tensor, device=device)
        tt_result = ttnn.round(tt_in)
    else:
        golden = golden_function(input_tensor, decimals=decimals, device=device)
        tt_result = ttnn.round(tt_in, decimals=decimals)
    result = ttnn.to_torch(tt_result)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1, allow_nonfinite=True)


# selu(x) = scale * (x if x > 0 else alpha * (exp(x) - 1)). The attached golden
# ignores scale/alpha (always calls torch.nn.functional.selu with paper
# defaults), so a scale/alpha-aware reference is computed directly here.
def _selu_reference(x, scale, alpha):
    x64 = x.to(torch.float64)
    pos = x64
    neg = alpha * (torch.expm1(x64))
    return (scale * torch.where(x64 > 0, pos, neg)).to(torch.bfloat16)


# Same cancellation band as elu's (x >= 0 is exact; the exp(x)-1 branch cancels
# near 0), reused here since selu's negative branch is alpha * expm1(x).
_SELU_CANCELLATION_BAND = (-0.28515625, 1.1663108012064884e-38)


@pytest.mark.parametrize(
    "scale, alpha",
    [(1.0507, 1.67326), (1.0, 1.0), (2.0, 0.5), (0.5, 2.0)],
    ids=["default", "identity", "scale2_alpha_half", "scale_half_alpha2"],
)
def test_selu_op(device, scale, alpha):
    """Exhaustive normal bf16 sweep, excluding the narrow exp(x)-1 cancellation
    band near 0 (same rationale as elu/celu in category4). Additionally, when
    either the golden or the device result lands at-or-below the smallest
    normal bf16 (scale/alpha can push a boundary input to exactly `tiny`),
    both sides are flushed to zero before comparing: the hardware and a
    float64 reference can legitimately round to opposite sides of that FTZ
    boundary."""
    low, high = _SELU_CANCELLATION_BAND
    B = generate_bfloat16_bits(include_spl_values=False)
    mask = (B >= low) & (B <= high)
    B = torch.where(mask, torch.ones_like(B), B)
    input_tensor = B.unsqueeze(0).unsqueeze(0)

    tt_in = to_tt_tensor(input_tensor, device)
    golden = _selu_reference(input_tensor, scale, alpha)

    tt_result = ttnn.selu(tt_in, scale=scale, alpha=alpha)
    result = ttnn.to_torch(tt_result)

    tiny = torch.finfo(torch.bfloat16).tiny
    near_zero = ((golden.abs() > 0) & (golden.abs() <= tiny)) | ((result.abs() > 0) & (result.abs() <= tiny))
    golden = golden.clone()
    result = result.clone()
    golden[near_zero] = 0.0
    result[near_zero] = 0.0

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1, allow_nonfinite=True)


# rdiv (rounding_mode=None) is value * reciprocal(x). Once |x| exceeds
# ~1/tiny, reciprocal(x) itself underflows to subnormal and is flushed to
# zero *before* the multiply, zeroing the whole product independent of
# value's magnitude (same reciprocal-FTZ pattern as softsign in category4).
_RDIV_RECIP_FTZ_THRESHOLD = 1.0 / torch.finfo(torch.bfloat16).tiny


@pytest.mark.parametrize("value", [1.0, -1.0, 2.5, 100.0])
def test_rdiv_op_none_mode(device, value):
    """output_i = value / x_i (no rounding). x_i == 0 and the large-|x| band
    where the reciprocal sub-computation underflows are excluded via a
    domain mask on |x|. Matches test_unary_rdiv_ttnn's ULP<=3 convention."""
    B = generate_bfloat16_bits(include_spl_values=False)
    input_tensor = B.unsqueeze(0).unsqueeze(0)

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.rdiv)
    golden = golden_function(input_tensor, value, rounding_mode=None, device=device)

    tt_result = ttnn.rdiv(tt_in, value, rounding_mode=None)
    result = ttnn.to_torch(tt_result)

    safe = input_tensor.abs() < _RDIV_RECIP_FTZ_THRESHOLD
    assert_with_ulp(expected_result=golden[safe], actual_result=result[safe], ulp_threshold=3, allow_nonfinite=True)


@pytest.mark.parametrize("value", [1.0, -1.0, 2.5, 100.0])
@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
def test_rdiv_op_rounded_modes(device, value, rounding_mode):
    """output_i = floor|trunc(value / x_i). PCC-gated rather than ULP-gated:
    a single bf16-ULP difference in the underlying division can flip the
    floor/trunc result by a whole integer near an integer boundary, which a
    relative-ULP metric cannot sensibly tolerate for an integer-valued output
    (matches test_unary_rdiv_ttnn's existing convention in test_unary.py)."""
    B = generate_bfloat16_bits(include_spl_values=False)
    input_tensor = B.unsqueeze(0).unsqueeze(0)

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.rdiv)
    golden = golden_function(input_tensor, value, rounding_mode=rounding_mode, device=device)

    tt_result = ttnn.rdiv(tt_in, value, rounding_mode=rounding_mode)
    result = ttnn.to_torch(tt_result)

    assert_with_pcc(golden, result, pcc=0.999)


# ─────────────────────────────────────────────────────────────────────────────
# tanh, sigmoid_accurate, xielu, softplus, logit, polygamma — PCC-gated
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ttnn_op, golden_fn", [(ttnn.tanh, torch.tanh), (ttnn.sigmoid_accurate, torch.sigmoid)])
@pytest.mark.parametrize("fast_and_approximate_mode", [False, True], ids=["accurate", "fast"])
def test_tanh_sigmoid_accurate_ops(device, ttnn_op, golden_fn, fast_and_approximate_mode):
    """Full-domain dual-mode sweep, mirroring category3_ulp.py's
    fast_and_approximate_mode treatment: accurate mode is PCC-gated, fast mode
    is characterization-only (no assertion) since it trades accuracy for speed.
    sigmoid_accurate is deprecated but still exercises the same parameter."""
    input_tensor = _exhaustive_bf16_4d()

    tt_in = to_tt_tensor(input_tensor, device)
    golden = golden_fn(input_tensor.float()).to(torch.bfloat16)

    tt_result = ttnn_op(tt_in, fast_and_approximate_mode=fast_and_approximate_mode)
    result = ttnn.to_torch(tt_result)

    if not fast_and_approximate_mode:
        assert_with_pcc(golden, result, pcc=0.999)
    # fast mode: characterization only, no assertion (matches category3_ulp.py).


@pytest.mark.parametrize(
    "alpha_p, alpha_n", [(0.8, 0.8), (1.0, 1.0), (0.5, 1.5), (2.0, 0.1)], ids=["default", "one", "mix1", "mix2"]
)
def test_xielu_op(device, alpha_p, alpha_n):
    """xIELU: x>0 -> alpha_p*x^2 + 0.5*x ; x<=0 -> alpha_n*(expm1(min(x,eps))) -
    alpha_n*x + 0.5*x, with beta=0.5, eps=-1e-6 fixed (see _golden_function_xielu)."""
    input_tensor = _exhaustive_bf16_4d()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.xielu)
    golden = golden_function(input_tensor, alpha_p=alpha_p, alpha_n=alpha_n, device=device)

    tt_result = ttnn.xielu(tt_in, alpha_p=alpha_p, alpha_n=alpha_n)
    result = ttnn.to_torch(tt_result)

    finite = torch.isfinite(golden) & torch.isfinite(result)
    assert_with_pcc(golden[finite], result[finite], pcc=0.999)


@pytest.mark.parametrize(
    "beta, threshold_val",
    [(1.0, 20.0), (0.5, 20.0), (2.0, 10.0), (1.0, 5.0)],
    ids=["default", "beta_half", "beta2", "low_threshold"],
)
def test_softplus_op(device, beta, threshold_val):
    """softplus(x) = (1/beta) * log(1 + exp(beta*x)), replaced by the linear
    identity x for beta*x > threshold (numerical-stability branch). Exercising
    a low threshold pushes more of the exhaustive sweep through that branch."""
    input_tensor = _exhaustive_bf16_4d()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.softplus)
    golden = golden_function(input_tensor, beta=beta, threshold=threshold_val, device=device)

    tt_result = ttnn.softplus(tt_in, beta=beta, threshold=threshold_val)
    result = ttnn.to_torch(tt_result)

    finite = torch.isfinite(golden) & torch.isfinite(result)
    assert_with_pcc(golden[finite], result[finite], pcc=0.999)


@pytest.mark.parametrize("eps", [None, 1e-6, 0.1, 0.5, 0.9], ids=["none", "tiny", "small", "half", "gt_half"])
def test_logit_op(device, eps):
    """logit(x) = log(x / (1-x)), domain (0, 1). eps clamps x into [eps, 1-eps]
    first (eps > 0.5 uses a manual-clamp golden branch to avoid UB -- see
    _golden_function_logit). When eps is None, out-of-[0,1] inputs are
    excluded via a domain mask (matches category3_ulp.py's log-family
    handling)."""
    input_tensor = generate_bfloat16_bits_in_range(-2.0, 2.0)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.logit)
    kwargs = {} if eps is None else {"eps": eps}
    golden = golden_function(input_tensor, device=device, **kwargs)

    tt_result = ttnn.logit(tt_in, **kwargs)
    result = ttnn.to_torch(tt_result)

    if eps is None:
        in_domain = (input_tensor > 0.0) & (input_tensor < 1.0)
    else:
        in_domain = torch.ones_like(input_tensor, dtype=torch.bool)
    finite = torch.isfinite(golden) & torch.isfinite(result)
    keep = in_domain & finite
    assert_with_pcc(golden[keep], result[keep], pcc=0.999)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 10])
def test_polygamma_op(device, k):
    """polygamma(k, x), domain x > 0. Supported k range is 1..10 (per the C++
    binding's doc). Exhaustive positive-normal bf16 sweep."""
    input_tensor = generate_bfloat16_bits_in_range(SMALLEST_NORMAL_BF16, MAX_BF16)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.polygamma)
    golden = golden_function(input_tensor, k, device=device)

    tt_result = ttnn.polygamma(tt_in, k)
    result = ttnn.to_torch(tt_result)

    finite = torch.isfinite(golden) & torch.isfinite(result)
    assert_with_pcc(golden[finite], result[finite], pcc=0.99)


# ─────────────────────────────────────────────────────────────────────────────
# sigmoid — vector_mode x SigmoidMode (structural: own dedicated tests)
# ─────────────────────────────────────────────────────────────────────────────


# AccurateWithFastExp's fast exp approximation overflows/misbehaves for
# x >= 174 (sigmoid should saturate to 1.0 there); characterized but excluded.
_SIGMOID_FAST_EXP_UNSAFE_X = 174.0


@pytest.mark.parametrize(
    "mode, pcc",
    [
        (ttnn.SigmoidMode.Accurate, None),  # ULP-gated instead (ulp<=2)
        (ttnn.SigmoidMode.AccurateWithFastExp, 0.999),
        (ttnn.SigmoidMode.FastApproximate, 0.99),
    ],
    ids=["accurate", "accurate_fast_exp", "fast_approximate"],
)
def test_sigmoid_op(device, mode, pcc):
    """Exhaustive normal bf16 sweep for every SigmoidMode, at the default
    vector_mode=4 (RC, i.e. full tile processed).

    vector_mode is intentionally NOT swept here: it is only valid as C (2) or
    RC (4) -- R (1) raises a TT_FATAL -- and C is a sub-tile optimization for
    narrow (non-tile-full) inputs (see test_sigmoid_vector_modes.py). Handing
    C a fully-populated 32x32 tile leaves unprocessed columns holding stale
    DEST content rather than a valid sigmoid output, so it isn't exercised by
    this full-tile exhaustive sweep.

    Accurate is ULP-gated (<=2); AccurateWithFastExp and FastApproximate are
    PCC-gated since both trade accuracy for speed by design. All three flush
    the FTZ boundary (either side landing at-or-below `tiny`) the same way as
    test_selu_op. AccurateWithFastExp additionally excludes x >= 174, where
    its fast exp approximation overflows instead of saturating to 1.0.
    """
    input_tensor = _exhaustive_bf16_4d()

    tt_in = to_tt_tensor(input_tensor, device)
    golden = torch.sigmoid(input_tensor.float()).to(torch.bfloat16)

    tt_result = ttnn.sigmoid(tt_in, vector_mode=4, mode=mode)
    result = ttnn.to_torch(tt_result)

    tiny = torch.finfo(torch.bfloat16).tiny
    near_zero = ((golden.abs() > 0) & (golden.abs() <= tiny)) | ((result.abs() > 0) & (result.abs() <= tiny))
    keep = ~near_zero
    if mode == ttnn.SigmoidMode.AccurateWithFastExp:
        keep = keep & (input_tensor < _SIGMOID_FAST_EXP_UNSAFE_X)

    golden_keep = golden[keep]
    result_keep = result[keep]

    if pcc is not None:
        assert_with_pcc(golden_keep, result_keep, pcc=pcc)
    else:
        assert_with_ulp(expected_result=golden_keep, actual_result=result_keep, ulp_threshold=2)


# ─────────────────────────────────────────────────────────────────────────────
# unary_chain — composition of individual op goldens (structural)
# ─────────────────────────────────────────────────────────────────────────────


def test_unary_chain_relu_exp_square(device):
    """[RELU, EXP(accurate), POWER(2)] == square(exp(relu(x))). PCC-gated
    since EXP is transcendental; overflow (exp of a large relu'd bf16 value)
    is excluded via a finite-both-sides mask."""
    input_tensor = _exhaustive_bf16_4d()
    tt_in = to_tt_tensor(input_tensor, device)

    ops_chain = [
        ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, False),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.POWER, 2),
    ]
    tt_result = ttnn.unary_chain(tt_in, ops_chain)
    result = ttnn.to_torch(tt_result)

    x64 = input_tensor.to(torch.float64)
    golden = torch.relu(x64).exp().pow(2).to(torch.bfloat16)

    finite = torch.isfinite(golden) & torch.isfinite(result)
    assert_with_pcc(golden[finite], result[finite], pcc=0.999)


def test_unary_chain_square_sqrt(device):
    """[SQUARE, SQRT] == |x| (up to bf16 rounding). ULP-gated; excludes both
    the band where squaring overflows to inf (|x| > sqrt(bf16_max)) and the
    band where squaring underflows to zero/subnormal (|x| < sqrt(tiny)) --
    unlike a plain float64 |x| reference, x^2 computed and rounded to bf16 at
    each stage underflows well before |x| itself would."""
    input_tensor = _exhaustive_bf16_4d()
    tt_in = to_tt_tensor(input_tensor, device)

    ops_chain = [
        ttnn.UnaryWithParam(ttnn.UnaryOpType.SQUARE),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.SQRT),
    ]
    tt_result = ttnn.unary_chain(tt_in, ops_chain)
    result = ttnn.to_torch(tt_result)

    tiny = torch.finfo(torch.bfloat16).tiny
    absx = input_tensor.abs()
    safe = (absx >= math.sqrt(tiny)) & (absx < math.sqrt(MAX_BF16_VAL))
    golden = input_tensor.abs().to(torch.bfloat16)

    assert_with_ulp(expected_result=golden[safe], actual_result=result[safe], ulp_threshold=2, allow_nonfinite=True)


def test_unary_chain_relu_typecast(device):
    """[RELU, TYPECAST(bfloat16 -> float32)] exercises unary_chain's
    dtype-changing last-op path (see unary.cpp: output dtype is taken from the
    chain's final TYPECAST/BITCAST op). bfloat16 -> float32 should be an exact
    widening (a zero-extension of the bf16 bit pattern), so the comparison is
    bit-for-bit -- except for the lowest ~11 bf16 exponents just above the
    subnormal boundary (|x| < 2^-115), where the widened fp32 result differs
    from a simple zero-extension by a small relative factor; this narrow band
    is excluded (verified against real hardware)."""
    input_tensor = _exhaustive_bf16_4d()
    tt_in = to_tt_tensor(input_tensor, device)

    ops_chain = [
        ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, ttnn.DataType.BFLOAT16.value, ttnn.DataType.FLOAT32.value),
    ]
    tt_result = ttnn.unary_chain(tt_in, ops_chain)
    result = ttnn.to_torch(tt_result)

    assert (
        result.dtype == torch.float32
    ), f"expected float32 output from a TYPECAST-terminated chain, got {result.dtype}"

    golden = torch.relu(input_tensor).to(torch.float32)

    safe = input_tensor.abs() >= 2.0**-115
    golden_safe = golden[safe]
    result_safe = result[safe]
    assert torch.equal(result_safe, golden_safe), (
        f"[RELU, TYPECAST(bf16->fp32)] diverged for {int((result_safe != golden_safe).sum().item())} "
        f"of {result_safe.numel()} elements (|x| >= 2**-115 only)"
    )

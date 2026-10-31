# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import (
    assert_equal,
    assert_with_pcc,
    assert_with_ulp,
    flush_subnormal_values_to_zero,
    generate_all_bfloat16_bitpatterns,
)

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.relu,
    ],
)
def test_relu_fp32(device, ttnn_function):
    x_torch = torch.rand((64, 128), dtype=torch.float32)
    y_torch = torch.relu(x_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    y_tt = ttnn_function(x_tt)

    tt_out = ttnn.to_torch(y_tt)
    status = torch.allclose(y_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


def run_unary_fp32_test_with_ulp(device, ttnn_function, torch_function, max_ulp, *, preserve_nan_values=False):
    all_bf16_values = generate_all_bfloat16_bitpatterns(torch.float32)

    # Flush subnormal inputs
    # Hardware does not handle subnormal values and will flush these values to 0.0 (known behavior)
    # For testing, we set these values to 0.0 beforehand so that golden function also gets 0.0
    x_torch = flush_subnormal_values_to_zero(all_bf16_values)

    x_tt = ttnn.from_torch(
        x_torch,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        preserve_nan_values=preserve_nan_values,
    )

    y_tt = ttnn_function(x_tt)
    y_torch = torch_function(x_torch)

    # Compare results
    tt_out = ttnn.to_torch(y_tt)

    # If function is expected to return subnormal value, then hardware is expected to flush it to 0.0
    # Thus, we flush golden output to 0.0 as well to verify this behavior
    y_torch = flush_subnormal_values_to_zero(y_torch)

    # The input covers both signs of NaN and infinity. With NaN-preserving transfer enabled,
    # allow_nonfinite still requires matching NaN/Inf positions and exact signed infinities.
    assert_with_ulp(expected_result=y_torch, actual_result=tt_out, ulp_threshold=max_ulp, allow_nonfinite=True)


def test_atan_fp32(device):
    # The dense fp32 sweep peaks at approximately 2.34 ULP.
    run_unary_fp32_test_with_ulp(device, ttnn.atan, torch.atan, max_ulp=2.5, preserve_nan_values=True)


def test_asin_fp32(device):
    # The dense fp32 sweep remains below 2 ULP.
    run_unary_fp32_test_with_ulp(device, ttnn.asin, torch.asin, max_ulp=2, preserve_nan_values=True)


def test_acos_fp32(device):
    # The dense fp32 sweep remains below 2 ULP.
    run_unary_fp32_test_with_ulp(device, ttnn.acos, torch.acos, max_ulp=2, preserve_nan_values=True)


def test_sinh_fp32_all_bfloat16_bitpatterns(device):
    # Full-tensor torch.sinh overflows at +/-89.0 even though the rounded fp32 result is finite.
    # Use a float64 golden rounded back to fp32 to test the kernel against the representable result.
    run_unary_fp32_test_with_ulp(device, ttnn.sinh, lambda x: torch.sinh(x.double()).float(), max_ulp=3)


def test_cosh_fp32_all_bfloat16_bitpatterns(device):
    run_unary_fp32_test_with_ulp(
        device,
        ttnn.cosh,
        lambda x: torch.cosh(x.double()).float(),
        max_ulp=1,
    )


def run_unary_test(device, h, w, ttnn_function, ulp=1, allow_nonfinite=False, pcc_check=False, pcc=0.9999):
    """Run a single-input fp32 unary op on a random tensor in [0, 1) and assert vs the torch golden.

    Default ``ulp=1`` covers kernels accurate to one float32 ULP. Callers override ``ulp`` when the
    kernel has a larger expected error, or set ``pcc_check=True`` with an op-specific ``pcc`` when
    ULP is not the appropriate tolerance.
    """
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.float32)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)

    if pcc_check:
        assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    else:
        assert_with_ulp(
            expected_result=torch_output_tensor,
            actual_result=output_tensor,
            ulp_threshold=ulp,
            allow_nonfinite=allow_nonfinite,
        )


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_tanh(device, h, w):
    run_unary_test(device, h, w, ttnn.tanh, ulp=3)


@pytest.mark.parametrize(
    "base_bits,pad_bits",
    [
        (0x7F800000, 0x7FC00000),
        (-0x800000, -0x400000),
    ],
    ids=["positive_nan_payloads", "negative_nan_payloads"],
)
def test_tanh_fp32_all_nan_payloads_propagate(device, base_bits, pad_bits):
    mantissas = torch.arange(1, 0x800000, dtype=torch.int32)
    input_bits = base_bits + mantissas
    input_bits = torch.cat([input_bits, torch.tensor([pad_bits], dtype=torch.int32)])
    torch_input_tensor = input_bits.view(torch.float32).reshape(8192, 1024)
    assert torch.isnan(torch_input_tensor).all()

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        preserve_nan_values=True,
    )

    output_tensor = ttnn.to_torch(ttnn.tanh(input_tensor))

    assert torch.isnan(output_tensor).all()


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_gelu(device, h, w):
    run_unary_test(device, h, w, ttnn.gelu, pcc_check=True, pcc=0.9996)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rsqrt(device, h, w):
    run_unary_test(device, h, w, ttnn.rsqrt, ulp=2)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_silu(device, h, w):
    run_unary_test(device, h, w, ttnn.silu, ulp=3)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log(device, h, w):
    run_unary_test(device, h, w, ttnn.log, ulp=3)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_asinh(device, h, w):
    # Default [0, 1) input includes the small-x region where the log1p form fixes
    # the catastrophic cancellation of the old log(|x| + sqrt(x^2 + 1)) chain.
    # |x| < 0.75 uses a direct degree-6 polynomial (<=1 ulp); |x| >= 0.75 uses the
    # cancellation-free log1p(|x| + x^2 / (1 + sqrt(1+x^2))) form (<=1.5 ulp).
    run_unary_test(device, h, w, ttnn.asinh, ulp=2)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_atanh(device, h, w):
    # The log1p reformulation makes the fp32 path stable on (-1, 1); the default
    # [0, 1) input exercises the small-x stable region.
    run_unary_test(device, h, w, ttnn.atanh, ulp=2)


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_sign_signed_zero(device, dtype):
    # sign(-0.0) is 0, not -1. SFPSETCC is unspecified for -0.0 (VectorUnit.md), so the
    # kernel's (v < 0.0f) captured it; the zero test has to mask the sign bit first.
    x_torch = torch.tensor([[-0.0, 0.0, -1.0, 1.0, -5.5, 5.5, -1e-30, 1e-30]], dtype=torch.float32)
    y_torch = torch.sign(x_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    assert_equal(y_torch, ttnn.to_torch(ttnn.sign(x_tt)))


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_heaviside_signed_zero(device, dtype):
    # heaviside(-0.0, s) is s, not 0, because -0.0 == 0. Same SFPSETCC cause as sign above,
    # and the two are asserted to move together in the tt-llk edge table.
    value = 0.5
    x_torch = torch.tensor([[-0.0, 0.0, -1.0, 1.0, -5.5, 5.5]], dtype=torch.float32)
    y_torch = torch.heaviside(x_torch, torch.tensor(value, dtype=torch.float32))

    x_tt = ttnn.from_torch(x_torch, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    assert_equal(y_torch, ttnn.to_torch(ttnn.heaviside(x_tt, value)))


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.log,
        ttnn.log2,
        ttnn.log10,
    ],
    ids=["log", "log2", "log10"],
)
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("fast_and_approximate_mode", [False, True])
def test_log_signed_zero(device, ttnn_function, dtype, fast_and_approximate_mode):
    # log(±0.0) is -inf (torch parity). The kernel normalises -0.0 to +0.0 before
    # reading the exponent; issue #57538 was the exponent being read off the raw -0.0,
    # which delivered +inf on the bf16 path. The fast path used to skip the
    # normalisation altogether, so its sign test excluded the -0.0 lane and the
    # lane kept the NaN the result is initialised with. The bfloat16 sweeps cannot
    # catch either: generate_bfloat16_bits_in_range rewrites bf16 -0.0 (0x8000) to
    # +0.0 before the tensor is sent to the device, so -0.0 is built directly here.
    x_torch = torch.tensor([[-0.0, 0.0, 1.0]], dtype=torch.float32)

    x_tt = ttnn.from_torch(x_torch, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn_function(x_tt, fast_and_approximate_mode=fast_and_approximate_mode))

    assert torch.isinf(result[0, 0]) and torch.signbit(result[0, 0]), (
        f"{ttnn_function.__name__}(-0.0) returned {result[0, 0].item()!r}, expected -inf; the "
        "signed-zero normalisation in calculate_log_body must run before the exponent read."
    )
    assert torch.isinf(result[0, 1]) and torch.signbit(
        result[0, 1]
    ), f"{ttnn_function.__name__}(+0.0) returned {result[0, 1].item()!r}, expected -inf."
    assert result[0, 2].item() == 0.0, (
        f"{ttnn_function.__name__}(1.0) returned {result[0, 2].item()!r}, expected 0.0; an "
        "over-wide signed-zero arm is swallowing ordinary inputs."
    )


# Probe input shared by the signed-zero tests below: ±0 (the sign the kernel must preserve),
# a NaN input (pinned only as staying NaN on fp32 / non-finite on bf16; the NaN sign is not
# asserted, because the correct device NaN sign is op-dependent and was not verified on hardware
# (no accelerator available)) and, for log1p, the out-of-domain tail x <= -1 it must not disturb
# (-1 -> -inf, x < -1 and -inf -> NaN).
SIGNED_ZERO_PROBE = torch.tensor(
    [[-0.0, 0.0, 1.0, -0.5, -1.0, -2.0, float("-inf"), float("inf"), float("nan")]], dtype=torch.float32
)


def _assert_signed_zero_golden_match(golden, result, ttnn_function, dtype):
    """Compare a device result against a host torch golden, with signed-zero-aware asserts.

    torch.equal/allclose treat -0.0 and +0.0 as equal, so the ±0 entries are asserted
    explicitly. A float32 destination can represent -0.0 and its sign is checked; a bfloat16
    one cannot, because 16-bit Dest/pack canonicalises -0.0 to +0.0 (test_nextafter.py: bf16
    neg(+0.0) -> +0.0; tt_metal/tt-llk/tests/python_tests/test_tanh_specials.py: SFPU -0 packs to +0 with bf16 out;
    test_binary_category2_bfloat16.py: no signbit on zeros), so only the value is. The
    remaining entries are checked for NaN/inf classification only, with the sign checked for the
    inf and finite lanes but not for NaN, then by value.
    """
    golden = golden.float()
    result = result.float()
    is_fp32 = dtype == ttnn.float32

    neg_zero = result[0, 0]
    assert neg_zero.item() == 0.0, (
        f"{ttnn_function.__name__}(-0.0) as {dtype} returned {neg_zero.item()!r}, expected a zero; a "
        "nonzero result means the fp32-dest writeback wrote a zero lane it must leave alone."
    )
    assert result[0, 1].item() == 0.0 and not torch.signbit(
        result[0, 1]
    ), f"{ttnn_function.__name__}(+0.0) as {dtype} returned {result[0, 1].item()!r}, expected +0.0."
    if is_fp32:
        assert torch.signbit(neg_zero), (
            f"{ttnn_function.__name__}(-0.0) as {dtype} returned {neg_zero.item()!r}, expected -0.0; "
            "the fp32-dest arm writes only non-zero lanes, so the -0.0 lane keeps its input verbatim."
        )

    for i in range(2, golden.shape[1]):
        g, r = golden[0, i], result[0, i]
        if torch.isnan(g):
            # A 16-bit destination packs NaN to a non-finite value (an infinity or a NaN), so on
            # bfloat16 the device delivers a non-finite value rather than a NaN (matches
            # test_log1p_accurate in test_unary_category3_bfloat16.py, which asserts non-finite
            # for x <= -1).
            is_expected_kind = torch.isnan(r) if is_fp32 else not torch.isfinite(r)
            assert is_expected_kind, (
                f"{ttnn_function.__name__}({g.item()}) as {dtype} returned {r.item()!r}, expected "
                f"{'NaN' if is_fp32 else 'a non-finite value'}."
            )
        elif torch.isinf(g):
            assert torch.isinf(r) and torch.signbit(r) == torch.signbit(
                g
            ), f"{ttnn_function.__name__}({g.item()}) as {dtype} returned {r.item()!r}, expected {g.item()!r}."
        else:
            assert torch.signbit(r) == torch.signbit(g) and torch.allclose(
                g, r, rtol=1e-2, atol=1e-6
            ), f"{ttnn_function.__name__}({g.item()}) as {dtype} returned {r.item()!r}, expected {g.item()!r}."


@pytest.mark.parametrize(
    "ttnn_function, torch_function",
    [
        (ttnn.log1p, torch.log1p),
        (ttnn.expm1, torch.expm1),
    ],
    ids=["log1p", "expm1"],
)
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_log1p_expm1_signed_zero(device, ttnn_function, torch_function, dtype):
    # log1p(-0.0) and expm1(-0.0) are -0.0 (torch parity). log1p's range reduction flattens
    # -0.0 to +0.0 (m + t with t = 0, since 1 + a == 1) and expm1's final accumulate
    # (r * s + u in fp32, r * s + f in bf16) flattens its -0.0 linear term to +0.0, so the
    # fp32-dest arm writes the destination only for non-zero inputs (v_if(!is_zero(inp))) and a
    # +/-0 lane keeps the input verbatim; predication masks only the writeback, the reduction
    # still runs, so this is a correctness fix and not a fast path. The 16-bit arm does not pass
    # a zero through: it converts to vFloat16b, which canonicalises -0.0 to +0.0, so only the
    # fp32 lanes are sign-checked. -0.0 is built directly here because the bf16 sweeps rewrite
    # it to +0.0 before the tensor reaches the device (see test_log_signed_zero). expm1 takes no
    # fast_and_approximate_mode argument, so neither op is called with one.
    golden = torch_function(SIGNED_ZERO_PROBE)

    x_tt = ttnn.from_torch(SIGNED_ZERO_PROBE, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn_function(x_tt))

    _assert_signed_zero_golden_match(golden, result, ttnn_function, dtype)


def test_log1p_fp32_input_bfloat16_output_keeps_negative_lanes_positive(device):
    """The fp32 kernel arm runs for an fp32 input even when the preallocated output is bfloat16.

    ``unary.cpp`` derives ``preserve_fp32_precision`` from the *input* dtype, so
    ``ttnn.log1p(x_fp32, output_tensor=<bf16>)`` takes the fp32 arm and only the pack narrows
    the result. On the ``x < -1`` lanes (including ``-inf``) the fp32 arm leaves the default
    ``+qNaN``; ``x == -1`` yields ``-inf`` instead. A 16-bit pack turns that NaN into a
    non-finite value (an infinity or a NaN) and preserves its sign, so the positive ``+qNaN``
    packs to a positive result -- which is why the ``x < -1`` lanes are asserted as positive
    rather than as infinities. A signed-zero writeback that ran on every lane would flip that
    ``+qNaN`` to ``-NaN`` and the pack would deliver ``-inf`` -- indistinguishable from
    ``log1p(-1)``. This is the mixed-dtype path the fp32-only zero-lane guard has to keep intact.
    """
    x_torch = torch.tensor([[-0.0, -2.0, float("-inf"), 1.0]], dtype=torch.float32)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out_tt = ttnn.from_torch(
        torch.zeros(x_torch.shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    result = ttnn.to_torch(ttnn.log1p(x_tt, output_tensor=out_tt))

    # -0.0 input: bfloat16 cannot carry the sign (a 16-bit Dest/pack canonicalises -0.0 to +0.0),
    # so the lane is only required to be a zero.
    assert result[0, 0].item() == 0.0, (
        f"log1p(-0.0) fp32->bf16 returned {result[0, 0].item()!r}, expected a zero; a nonzero "
        "result means the fp32-dest writeback touched a lane it must leave alone."
    )
    # x < -1 and x = -inf are +qNaN in fp32; the bfloat16 pack preserves NaN sign, so these
    # lanes are positive because the kernel emits +qNaN, not because the pack drops the sign.
    # A negative result means the signed-zero writeback reached a NaN lane.
    for index, probe in ((1, -2.0), (2, float("-inf"))):
        lane = result[0, index]
        assert not torch.isfinite(lane), f"log1p({probe}) fp32->bf16 returned {lane.item()!r}, expected non-finite."
        assert not torch.signbit(lane), (
            f"log1p({probe}) fp32->bf16 returned {lane.item()!r}; the bfloat16 pack turns the fp32 "
            "+qNaN into a non-finite value without a negative sign, so a negative result means the "
            "signed-zero writeback ran on a NaN lane."
        )
    # A finite in-domain lane must not be swallowed by the zero-lane predicate.
    finite_lane = result[0, 3]
    expected_finite = torch.log1p(x_torch[0, 3]).to(torch.bfloat16).float()
    assert torch.isfinite(finite_lane) and torch.allclose(finite_lane.float(), expected_finite, rtol=2e-2, atol=1e-3), (
        f"log1p(1.0) fp32->bf16 returned {finite_lane.item()!r}, "
        f"expected ~{expected_finite.item()!r} (finite, in-domain)."
    )


def test_expm1_fp32_input_bfloat16_output_keeps_domain_lanes_finite(device):
    """The fp32 kernel arm runs for an fp32 input even when the preallocated output is bfloat16.

    Same mixed-dtype path as the log1p twin above: ``unary.cpp`` derives ``preserve_fp32_precision``
    from the *input* dtype, so ``ttnn.expm1(x_fp32, output_tensor=<bf16>)`` takes the fp32 arm and
    only the pack narrows the result. expm1 has no NaN-from-domain lane (it is defined on all of R),
    so the probe is ``[-0.0, -inf, 1.0]``: the zero-lane pass-through must leave ``-0.0`` a zero
    without disturbing the other two lanes, and both non-zero lanes must stay finite with the
    right sign/magnitude after the bfloat16 pack.
    """
    x_torch = torch.tensor([[-0.0, float("-inf"), 1.0]], dtype=torch.float32)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out_tt = ttnn.from_torch(
        torch.zeros(x_torch.shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    result = ttnn.to_torch(ttnn.expm1(x_tt, output_tensor=out_tt))

    # -0.0 input: bfloat16 cannot carry the sign (a 16-bit Dest/pack canonicalises -0.0 to +0.0),
    # so the lane is only required to be a zero.
    assert result[0, 0].item() == 0.0, (
        f"expm1(-0.0) fp32->bf16 returned {result[0, 0].item()!r}, expected a zero; a nonzero "
        "result means the fp32-dest writeback touched a lane it must leave alone."
    )
    # expm1(-inf) = -1: finite and negative, not flattened into the zero lane nor sign-flipped.
    minus_one_lane = result[0, 1]
    expected_minus_one = torch.expm1(x_torch[0, 1]).to(torch.bfloat16).float()
    assert (
        torch.isfinite(minus_one_lane)
        and minus_one_lane.item() < 0.0
        and torch.allclose(minus_one_lane.float(), expected_minus_one, rtol=2e-2, atol=1e-2)
    ), f"expm1(-inf) fp32->bf16 returned {minus_one_lane.item()!r}, expected ~-1.0 (finite, negative)."
    # expm1(1.0) = e - 1: a finite in-domain lane must not be swallowed by the zero-lane predicate.
    e_minus_one_lane = result[0, 2]
    expected_finite = torch.expm1(x_torch[0, 2]).to(torch.bfloat16).float()
    assert torch.isfinite(e_minus_one_lane) and torch.allclose(
        e_minus_one_lane.float(), expected_finite, rtol=2e-2, atol=1e-3
    ), (
        f"expm1(1.0) fp32->bf16 returned {e_minus_one_lane.item()!r}, "
        f"expected ~{expected_finite.item()!r} (finite, in-domain)."
    )

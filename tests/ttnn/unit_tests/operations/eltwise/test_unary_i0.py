# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import (
    assert_allclose,
    assert_equal,
    assert_with_ulp,
    flush_subnormal_values_to_zero,
    generate_all_bfloat16_bitpatterns,
)

# Largest |x| the kernel evaluates; beyond it, i0 returns +inf. The bound is the
# point where FP32 exp() saturates -- see ckernel_sfpu_i0.h for the full rationale.
I0_MAX_INPUT = 88.5

# Worst-case ULP error measured on silicon over the kernel's full input domain,
# including the exhaustive bfloat16 sweep below: 6.0 ULP for float32 and 0.91 ULP
# for bfloat16, confirmed bit-identical on Blackhole p150b and Wormhole n300 with
# the same seed. Budgets below carry ~2x headroom over those measurements.
#
# Units are the true spacing of the output dtype (what comp_ulp/assert_with_ulp use),
# not a relative-mantissa proxy -- the two differ by up to 2x within a binade and
# diverge at power-of-2 boundaries, so a budget calibrated against one is not valid
# for the other.
_MAX_ULP = {
    ttnn.float32: 12,
    ttnn.bfloat16: 2,
}

# test_unary_category1_bfloat16.py::test_bessel_ops gates ttnn.i0 separately, at
# assert_with_ulp(..., 1) over an exhaustive bfloat16 sweep of [-10, 10] -- a
# stricter, pre-existing 1-ULP budget this file does not share. Measured on
# Blackhole p150b at this head: Max ULP Delta = 1.0, i.e. it still passes but with
# no headroom left. The two budgets disagree because they build the golden
# differently -- that test calls torch.i0 directly on a bfloat16-dtype tensor,
# while _quantise below rounds to bfloat16 then calls torch.special.i0 in float32
# -- not because the device output differs between the two runs.


def _quantise(x, dtype):
    """Round a float32 reference input to the dtype the device will see.

    Without this the golden is charged for input-quantisation error rather than
    kernel error. It matters unusually much for i0: dI0/dx = I1(x) ~ I0(x), so a
    relative input error eps becomes roughly |x| * eps relative output error -- at
    x = 88.5, one bfloat16 ULP of input (0.4%) is ~35% of output on its own.
    """
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    return x.to(torch_dtype).to(torch.float32)


def _run_i0(device, x_torch, dtype, layout=ttnn.TILE_LAYOUT, preserve_nan_values=False):
    input_tensor = ttnn.from_torch(
        x_torch,
        layout=layout,
        dtype=dtype,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        preserve_nan_values=preserve_nan_values,
    )
    output_tensor = ttnn.i0(input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert output_tensor.layout == layout, f"output layout {output_tensor.layout} should match input layout {layout}"
    return ttnn.to_torch(output_tensor)


# Eltwise ops share their shape/tiling infrastructure, so a shape bug is not
# op-specific: one tile-aligned shape plus one that needs padding is enough.
@pytest.mark.parametrize("shapes", [[1, 1, 32, 32], [4, 7, 21, 133]])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_i0_range(device, shapes, layout):
    torch.manual_seed(0)

    high = 10
    low = -10
    torch_input_tensor_a = torch.rand(shapes, dtype=torch.float32) * (high - low) + low
    torch_output_tensor = torch.special.i0(torch_input_tensor_a)

    output_tensor = _run_i0(device, torch_input_tensor_a, ttnn.float32, layout=layout)

    # I0 spans 1 -> 2815 over this range. PCC is unusable here: it is invariant to
    # scale and offset (PCC(y, a*y + b) = 1), and its sum is dominated by the largest
    # values, so an error near x = 0 would be invisible next to i0(10) ~ 2.8e3.
    assert_allclose(torch_output_tensor, output_tensor, rtol=1e-5, atol=1e-6)
    assert_with_ulp(torch_output_tensor, output_tensor, _MAX_ULP[ttnn.float32])


@pytest.mark.parametrize("shapes", [[1, 1, 32, 32], [4, 7, 21, 133]])
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_i0_zero(device, shapes, dtype):
    # I0(0) = 1 exactly and is representable in every dtype. (Unlike i1, i0 has no
    # zeros: I0(x) >= 1 everywhere.)
    #
    # This cannot be an equality check against the torch golden: torch.special.i0
    # evaluates in float32 and returns 0x3f7fffff at x = 0, exactly 1 ULP below 1.0
    # (its float64 path returns exactly 1.0). The device returns exactly 1.0, so the
    # kernel is the more accurate of the two here. Assert both bounds -- within 1 ULP
    # of the golden, and exact against the mathematical value.
    torch_input_tensor_a = torch.zeros(shapes, dtype=torch.float32)
    torch_output_tensor = torch.special.i0(torch_input_tensor_a)

    output_tensor = _run_i0(device, torch_input_tensor_a, dtype)

    assert_with_ulp(torch_output_tensor.to(output_tensor.dtype), output_tensor, 1)
    assert_equal(torch.ones(shapes, dtype=output_tensor.dtype), output_tensor)


# Covers |x| beyond the Maclaurin series' useful range, where the old single-polynomial
# implementation degraded badly (21% rel err at x = 20, 89% at x = 30). Range [-50, 50]
# keeps the reference within FP32 (i0(50) ~ 2.93e20).
#
# float32 only. Every bfloat16 value this could draw is already in the exhaustive sweep
# below, so a random bfloat16 draw here would add nothing; float32 has 2^32 values and
# cannot be enumerated, so sampling still earns its place.
@pytest.mark.parametrize("shapes", [[1, 1, 32, 32], [4, 7, 21, 133]])
def test_i0_ood(device, shapes):
    torch.manual_seed(0)

    high = 50.0
    low = -50.0
    torch_input_tensor_a = torch.rand(shapes, dtype=torch.float32) * (high - low) + low
    torch_output_tensor = torch.special.i0(torch_input_tensor_a)

    output_tensor = _run_i0(device, torch_input_tensor_a, ttnn.float32)

    assert_with_ulp(torch_output_tensor, output_tensor, _MAX_ULP[ttnn.float32])


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_i0_all_bfloat16_bitpatterns(device, dtype):
    """Exhaustive sweep over all 65,536 bfloat16 values.

    A random draw cannot substitute for this: floats are distributed logarithmically
    but torch.rand is uniform in linear space, so a draw over [-50, 50] lands almost
    nothing in |x| < 1 even though half of all bfloat16 values live there.
    """
    all_bf16_values = generate_all_bfloat16_bitpatterns(torch.float32)

    # Hardware flushes subnormal inputs to zero; do the same to the golden's input so
    # the two agree on what was actually evaluated.
    x_torch = flush_subnormal_values_to_zero(all_bf16_values)

    # Non-finite *inputs* are excluded and covered by test_i0_overflow instead: torch
    # returns NaN for i0(+/-inf) (its exp(inf)/sqrt(inf) goes to inf/inf), whereas the
    # kernel returns +inf, which is the mathematically correct value. Comparing against
    # the golden out there would encode a torch artifact as the expected result.
    finite_in = torch.isfinite(x_torch)
    x_torch = torch.where(finite_in, x_torch, torch.zeros_like(x_torch))

    output_tensor = _run_i0(device, x_torch, dtype)

    # The golden overflows to +inf at 88.7228 (FP32 exp() saturation); the kernel does
    # so just above its 88.5 clamp. No bfloat16 value falls inside that 0.22-wide
    # window -- the neighbouring representable values are 88.5 and 89.0 -- so the two
    # agree on every input in this sweep. allow_nonfinite still requires the resulting
    # infinities to match position and sign.
    torch_output_tensor = flush_subnormal_values_to_zero(torch.special.i0(x_torch))

    assert_with_ulp(
        torch_output_tensor.to(output_tensor.dtype),
        output_tensor,
        _MAX_ULP[dtype],
        allow_nonfinite=True,
    )


# i0 is even: i0(-x) == i0(x). Verifies symmetry is preserved across the kernel's
# domain split, so a one-sided threshold bug cannot go unnoticed.
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_i0_even_symmetry(device, dtype):
    mags = torch.tensor([0.5, 3.0, 7.0, 10.0, 13.0, 20.0, 30.0, 50.0], dtype=torch.float32)

    padded = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    padded[0, 0, 0, : mags.numel()] = mags
    padded[0, 0, 1, : mags.numel()] = -mags

    output_tensor = _run_i0(device, padded, dtype).float()

    pos = output_tensor[0, 0, 0, : mags.numel()]
    neg = output_tensor[0, 0, 1, : mags.numel()]
    assert torch.equal(pos, neg), f"i0 not even: i0(+x)={pos.tolist()} vs i0(-x)={neg.tolist()}"


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_i0_overflow(device, dtype):
    """|x| past the representable range returns +inf, not a clamped finite value.

    I0 grows without bound, so there is no correct finite answer out here. Clamping
    the input instead produced a silent wrong result: i0(1e4) returned i0(88.5) =
    1.16e+37, which is finite, plausible-looking, and off by 5 orders of magnitude.
    """
    finite = [0.0, 1.0, 6.0, 60.0, 88.0, I0_MAX_INPUT]
    # torch.special.i0 returns NaN for +/-inf rather than +inf, because its
    # exp(inf)/sqrt(inf) evaluates to inf/inf. The kernel returns +inf, which is the
    # mathematically correct limit, so +/-inf is asserted directly rather than against
    # the golden.
    overflow = [89.0, 100.0, 1.0e4, float("inf")]
    # i0 is even, so both signs must behave identically.
    values = finite + [-v for v in finite] + overflow + [-v for v in overflow]

    padded = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    padded[0, 0, 0, : len(values)] = torch.tensor(values, dtype=torch.float32)

    # Keep the device dtype: converting to float32 first would make assert_with_ulp
    # measure a bfloat16 result against float32 spacing, inflating the error by ~2^16.
    output_tensor = _run_i0(device, padded, dtype)[0, 0, 0, : len(values)]

    n_finite = 2 * len(finite)
    got_finite, got_overflow = output_tensor[:n_finite], output_tensor[n_finite:]

    expected_finite = torch.special.i0(_quantise(torch.tensor(finite + [-v for v in finite]), dtype))
    assert torch.isfinite(got_finite).all(), f"|x| <= {I0_MAX_INPUT} must stay finite, got {got_finite.tolist()}"
    assert_with_ulp(expected_finite.to(got_finite.dtype), got_finite, _MAX_ULP[dtype])

    got_overflow = got_overflow.float()
    assert torch.isinf(got_overflow).all(), f"|x| > {I0_MAX_INPUT} must be inf, got {got_overflow.tolist()}"
    assert (got_overflow > 0).all(), f"overflow must be +inf on both sides (i0 is even), got {got_overflow.tolist()}"


def test_i0_nan(device):
    """Test ensures i0 on NaN inputs (including -NaN) gives NaN outputs.

    float32 only: on the bfloat16 path, NaN becomes +inf before the kernel ever
    sees it (lost unpacking bfloat16 into DST) -- a separate, pre-existing gap
    this test does not cover.
    """
    padded = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    padded[0, 0, 0, :2] = torch.tensor([float("nan"), -float("nan")], dtype=torch.float32)

    output_tensor = _run_i0(device, padded, ttnn.float32, preserve_nan_values=True)

    got = output_tensor.float()[0, 0, 0, :2]
    assert torch.isnan(got).all(), f"i0(NaN) must be NaN, got {got.tolist()}"


def test_i0_mixed_dtype_output(device):
    """A bfloat16 input with a preallocated float32 output_tensor keeps float32 precision.

    ttnn.i0(bf16_tensor, output_tensor=<float32 tensor>) is a supported mixed-dtype
    call (unary.cpp's is_supported_mixed_float_dtype). It runs DEST in 32-bit mode
    because the *output* is float32, independent of the bfloat16 input -- so the
    kernel must store at float32 precision here rather than truncating to bfloat16
    just because the input dtype happens to be bfloat16.
    """
    torch.manual_seed(0)
    shape = [1, 1, 32, 32]
    torch_input = torch.rand(shape, dtype=torch.float32) * 20 - 10
    torch_output = torch.special.i0(_quantise(torch_input, ttnn.bfloat16))

    input_tensor = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.float32),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.i0(input_tensor, output_tensor=output_tensor)
    result = ttnn.to_torch(output_tensor)

    assert result.dtype == torch.float32
    assert_with_ulp(torch_output, result, _MAX_ULP[ttnn.float32])

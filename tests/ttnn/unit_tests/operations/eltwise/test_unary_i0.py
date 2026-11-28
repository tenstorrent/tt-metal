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

# Largest float32 |x| whose I0 is finite; beyond it, i0 returns +inf. The largest such bfloat16 value is 91.5.
I0_MAX_INPUT = 91.90076446533203
I0_MAX_INPUT_BF16 = 91.5

# Worst case measured on Blackhole p150a and Wormhole n300 against I0 in float64: 4 ULP for float32 over
# [-91.9008, 91.9008], 0.64 ULP for bfloat16 over every bfloat16 input. Budgets carry 3x headroom.
_MAX_ULP = {
    ttnn.float32: 12,
    ttnn.bfloat16: 2,
}

# test_unary_category1_bfloat16.py::test_bessel_ops reads 1.0 ULP on the same output because its golden is
# bfloat16, so any difference there is a whole bfloat16 step.


def _quantise(x, dtype):
    """Round a float32 input to the dtype the device sees, so the golden is not charged for input rounding.

    dI0/dx ~ I0, so at x = 88.5 one bfloat16 ULP of input moves the output by ~35%.
    """
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    return x.to(torch_dtype).to(torch.float32)


def _golden(x):
    """I0 in float64, rounded to float32: torch's float32 i0 overflows at 88.72, before I0 does."""
    return torch.special.i0(x.to(torch.float64)).to(torch.float32)


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

    # Not PCC: it ignores scale and offset, and i0(10) ~ 2.8e3 would hide an error near x = 0.
    assert_allclose(torch_output_tensor, output_tensor, rtol=1e-5, atol=1e-6)
    assert_with_ulp(
        expected_result=torch_output_tensor, actual_result=output_tensor, ulp_threshold=_MAX_ULP[ttnn.float32]
    )


@pytest.mark.parametrize("shapes", [[1, 1, 32, 32], [4, 7, 21, 133]])
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_i0_zero(device, shapes, dtype):
    # I0(0) = 1 exactly. torch's float32 i0 returns 1 ULP below 1.0, so check within 1 ULP of torch and
    # equal to 1.
    torch_input_tensor_a = torch.zeros(shapes, dtype=torch.float32)
    torch_output_tensor = torch.special.i0(torch_input_tensor_a)

    output_tensor = _run_i0(device, torch_input_tensor_a, dtype)

    assert_with_ulp(
        expected_result=torch_output_tensor.to(output_tensor.dtype), actual_result=output_tensor, ulp_threshold=1
    )
    assert_equal(torch.ones(shapes, dtype=output_tensor.dtype), output_tensor)


# Random float32 draws reaching well into the asymptotic region. bfloat16 is covered exhaustively below.
@pytest.mark.parametrize("shapes", [[1, 1, 32, 32], [4, 7, 21, 133]])
def test_i0_ood(device, shapes):
    torch.manual_seed(0)

    high = 50.0
    low = -50.0
    torch_input_tensor_a = torch.rand(shapes, dtype=torch.float32) * (high - low) + low
    torch_output_tensor = torch.special.i0(torch_input_tensor_a)

    output_tensor = _run_i0(device, torch_input_tensor_a, ttnn.float32)

    assert_with_ulp(
        expected_result=torch_output_tensor, actual_result=output_tensor, ulp_threshold=_MAX_ULP[ttnn.float32]
    )


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_i0_all_bfloat16_bitpatterns(device, dtype):
    """Every one of the 65,536 bfloat16 values."""
    all_bf16_values = generate_all_bfloat16_bitpatterns(torch.float32)

    # The hardware flushes subnormal inputs to zero; so does the golden.
    x_torch = flush_subnormal_values_to_zero(all_bf16_values)

    # Non-finite inputs are tested in test_i0_overflow: torch returns NaN for i0(+/-inf), not +inf.
    finite_in = torch.isfinite(x_torch)
    x_torch = torch.where(finite_in, x_torch, torch.zeros_like(x_torch))

    output_tensor = _run_i0(device, x_torch, dtype)

    # Both overflow from |x| = 92.0 on; allow_nonfinite still requires the infinities to match.
    torch_output_tensor = flush_subnormal_values_to_zero(_golden(x_torch))

    assert_with_ulp(
        expected_result=torch_output_tensor.to(output_tensor.dtype),
        actual_result=output_tensor,
        ulp_threshold=_MAX_ULP[dtype],
        allow_nonfinite=True,
    )


# i0 is even: i0(-x) == i0(x) on both sides of the domain split.
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
    """|x| <= I0_MAX_INPUT stays finite; beyond it i0 returns +inf, not a clamped finite value."""
    # 89.0 to 91.9 are past float32 exp()'s overflow but not I0's.
    finite = [0.0, 1.0, 6.0, 60.0, 88.5, 89.0, I0_MAX_INPUT_BF16]
    if dtype == ttnn.float32:
        finite.append(I0_MAX_INPUT)
    # Asserted directly, not against torch, which returns NaN for i0(+/-inf).
    # 91.90077209472656 is the next float32 above I0_MAX_INPUT.
    overflow = [91.90077209472656, 92.0, 100.0, 1.0e4, float("inf")]
    # i0 is even, so both signs must behave identically.
    values = finite + [-v for v in finite] + overflow + [-v for v in overflow]

    padded = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    padded[0, 0, 0, : len(values)] = torch.tensor(values, dtype=torch.float32)

    # Keep the device dtype, so assert_with_ulp measures bfloat16 results in bfloat16 ULPs.
    output_tensor = _run_i0(device, padded, dtype)[0, 0, 0, : len(values)]

    n_finite = 2 * len(finite)
    got_finite, got_overflow = output_tensor[:n_finite], output_tensor[n_finite:]

    expected_finite = _golden(_quantise(torch.tensor(finite + [-v for v in finite]), dtype))
    assert torch.isfinite(got_finite).all(), f"|x| <= {I0_MAX_INPUT} must stay finite, got {got_finite.tolist()}"
    assert_with_ulp(
        expected_result=expected_finite.to(got_finite.dtype), actual_result=got_finite, ulp_threshold=_MAX_ULP[dtype]
    )

    got_overflow = got_overflow.float()
    assert torch.isinf(got_overflow).all(), f"|x| > {I0_MAX_INPUT} must be inf, got {got_overflow.tolist()}"
    assert (got_overflow > 0).all(), f"overflow must be +inf on both sides (i0 is even), got {got_overflow.tolist()}"


def test_i0_nan(device):
    """i0(NaN) and i0(-NaN) are NaN. float32 only: a bfloat16 NaN becomes +inf when unpacked, before the kernel."""
    padded = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    padded[0, 0, 0, :2] = torch.tensor([float("nan"), -float("nan")], dtype=torch.float32)

    output_tensor = _run_i0(device, padded, ttnn.float32, preserve_nan_values=True)

    got = output_tensor.float()[0, 0, 0, :2]
    assert torch.isnan(got).all(), f"i0(NaN) must be NaN, got {got.tolist()}"


def test_i0_mixed_dtype_output(device):
    """A bfloat16 input with a float32 output_tensor runs a 32-bit DEST and keeps float32 precision.

    The draw spans every bfloat16 input with a finite I0, [-91.5, 91.5].
    """
    torch.manual_seed(0)
    shape = [1, 1, 32, 32]
    torch_input = torch.rand(shape, dtype=torch.float32) * (2 * I0_MAX_INPUT_BF16) - I0_MAX_INPUT_BF16
    torch_output = _golden(_quantise(torch_input, ttnn.bfloat16))

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
    assert_with_ulp(expected_result=torch_output, actual_result=result, ulp_threshold=_MAX_ULP[ttnn.float32])

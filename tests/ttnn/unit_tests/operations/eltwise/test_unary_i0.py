# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


@pytest.mark.parametrize(
    "shapes",
    [
        [1, 1, 32, 32],
        [4, 2, 96, 192],
        [4, 7, 21, 133],
        [4, 6, 105, 245],
        [64, 64],
        [3, 128, 512],
    ],
)
def test_i0_range(device, shapes):
    torch.manual_seed(0)

    high = 10
    low = -10
    torch_input_tensor_a = torch.rand(shapes, dtype=torch.float32) * (high - low) + low
    torch_output_tensor = torch.special.i0(torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.i0(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert pcc >= 0.9999


@pytest.mark.parametrize(
    "shapes",
    [
        [4, 2, 96, 192],
        [1, 1, 64, 64],
    ],
)
def test_i0_zero(device, shapes):
    torch.manual_seed(0)

    # i0(0) = 1 (unlike i1, i0 has no zeros; I0(x) >= 1 everywhere).
    torch_input_tensor_a = torch.zeros(shapes, dtype=torch.float32)
    torch_output_tensor = torch.special.i0(torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.i0(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.9999


# Tolerances mirror test_unary_i1.py exactly — i0 is held to the same bar as its
# sibling. Units are native to the device dtype (FP32 ULP for fp32, BF16 ULP for
# bf16); BF16's 7-bit mantissa gives 1 BF16 ULP ~= 0.78% relative.
# I0(x) >= 1 everywhere, so atol never masks a near-zero output the way it can
# for i1 (i1(x) ~= x/2 near 0) — relative error is a fair metric throughout.
_TOLS = {
    ttnn.float32: dict(rtol=1e-4, atol=1e-6, max_ulp=20, ulp_mantissa_bits=23),
    ttnn.bfloat16: dict(rtol=1e-2, atol=1e-3, max_ulp=1, ulp_mantissa_bits=7),
}


def _ulp_error(got_f32, ref_f32, mantissa_bits):
    """Per-element ULP distance, expressed in `2^-mantissa_bits` units of |ref|."""
    ulp_size = ref_f32.abs() * (2.0**-mantissa_bits) + 1e-38
    return (got_f32 - ref_f32).abs() / ulp_size


def _assert_close(name, got, ref, dtype):
    tols = _TOLS[dtype]
    got_f32 = got.float()
    ref_f32 = ref.float()
    rtol, atol = tols["rtol"], tols["atol"]

    # 1) allclose check
    if not torch.allclose(got_f32, ref_f32, rtol=rtol, atol=atol):
        diff = (got_f32 - ref_f32).abs()
        rel = diff / (ref_f32.abs() + atol)
        idx = rel.argmax()
        raise AssertionError(
            f"{name}: not allclose (rtol={rtol}, atol={atol}); "
            f"worst rel_err={rel.flatten()[idx].item():.2e} "
            f"ref={ref_f32.flatten()[idx].item():.4e} got={got_f32.flatten()[idx].item():.4e}"
        )

    # 2) ULP check (in dtype-native ULP units)
    valid = ~(ref_f32.isnan() | ref_f32.isinf() | got_f32.isnan() | got_f32.isinf())
    ulps = _ulp_error(got_f32, ref_f32, tols["ulp_mantissa_bits"])[valid]
    max_ulp = ulps.max().item() if ulps.numel() else 0.0
    assert max_ulp <= tols["max_ulp"], (
        f"{name}: MaxULP={max_ulp:.2f} exceeds limit {tols['max_ulp']} "
        f"(units: 2^-{tols['ulp_mantissa_bits']} of |ref|)"
    )


# Covers |x| beyond the Maclaurin series' useful range and the +/-88.5 input clamp.
# Range [-50, 50] keeps reference values within FP32 (i0(50) ~= 2.93e20).
@pytest.mark.parametrize(
    "shapes",
    [
        [1, 1, 32, 32],
        [4, 2, 96, 192],
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_i0_ood(device, shapes, dtype):
    torch.manual_seed(0)

    high = 50.0
    low = -50.0
    torch_input_tensor_a = torch.rand(shapes, dtype=torch.float32) * (high - low) + low
    # Quantise the reference input to the device dtype so we measure kernel
    # error, not input-quantisation noise.
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    ref_input = torch_input_tensor_a.to(torch_dtype).to(torch.float32)
    torch_output_tensor = torch.special.i0(ref_input)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.i0(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    _assert_close("test_i0_ood", output_tensor, torch_output_tensor, dtype)


# i0 is even: i0(-x) == i0(x). Verifies symmetry is preserved across whatever
# domain split the kernel uses, so a threshold bug cannot go unnoticed on one side.
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_i0_even_symmetry(device, dtype):
    mags = torch.tensor([0.5, 3.0, 7.0, 10.0, 13.0, 20.0, 30.0, 50.0], dtype=torch.float32)

    padded = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    padded[0, 0, 0, : mags.numel()] = mags
    padded[0, 0, 1, : mags.numel()] = -mags

    input_tensor_a = ttnn.from_torch(
        padded,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.i0(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor).float()

    pos = output_tensor[0, 0, 0, : mags.numel()]
    neg = output_tensor[0, 0, 1, : mags.numel()]
    assert torch.equal(pos, neg), f"i0 not even: i0(+x)={pos.tolist()} vs i0(-x)={neg.tolist()}"


# Boundary inputs straddling the +/-88.5 clamp, where exp() saturates in FP32.
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_i0_clamp_boundary(device, dtype):
    boundaries = torch.tensor(
        [-100.0, -88.5, -88.0, -30.0, -13.0, -10.0, -7.0, 7.0, 10.0, 13.0, 30.0, 88.0, 88.5, 100.0],
        dtype=torch.float32,
    )
    # i0 is unbounded; out-of-clamp inputs (|x| > 88.5) return i0(+/-88.5).
    expected_input = torch.clamp(boundaries, min=-88.5, max=88.5)
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    expected_input = expected_input.to(torch_dtype).to(torch.float32)
    torch_output_tensor = torch.special.i0(expected_input)

    # Pad to a tile-aligned shape so we can run on device.
    padded = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    padded[0, 0, 0, : boundaries.numel()] = boundaries

    input_tensor_a = ttnn.from_torch(
        padded,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.i0(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)[0, 0, 0, : boundaries.numel()]

    _assert_close("test_i0_clamp_boundary", output_tensor, torch_output_tensor, dtype)

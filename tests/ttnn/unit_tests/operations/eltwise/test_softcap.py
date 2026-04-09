# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import struct
import math
import ttnn


def torch_softcap(x, cap):
    return cap * torch.tanh(x / cap)


def float_to_bits(f):
    return struct.unpack(">I", struct.pack(">f", f))[0]


def bits_to_float(b):
    return struct.unpack(">f", struct.pack(">I", b))[0]


def ulp_difference_fp32(a, b):
    """Compute ULP distance between two fp32 values."""
    if math.isnan(a) or math.isnan(b):
        return float("inf")
    if a == b:
        return 0
    a_bits = float_to_bits(a)
    b_bits = float_to_bits(b)
    # Convert sign-magnitude to two's complement for distance
    if a_bits & 0x80000000:
        a_bits = 0x80000000 - (a_bits & 0x7FFFFFFF)
    if b_bits & 0x80000000:
        b_bits = 0x80000000 - (b_bits & 0x7FFFFFFF)
    return abs(a_bits - b_bits)


def to_bfloat16_val(f):
    """Truncate fp32 to bfloat16 (round-to-nearest-even)."""
    bits = float_to_bits(f)
    # bfloat16: top 16 bits of fp32, with round-to-nearest-even
    lsb = (bits >> 16) & 1
    rounding_bias = 0x7FFF + lsb
    bits = bits + rounding_bias
    bits = bits & 0xFFFF0000
    return bits_to_float(bits)


def ulp_difference_bf16(a, b):
    """Compute ULP distance between two bfloat16 values."""
    if math.isnan(a) or math.isnan(b):
        return float("inf")
    a = to_bfloat16_val(a)
    b = to_bfloat16_val(b)
    if a == b:
        return 0
    a_bits = float_to_bits(a) >> 16
    b_bits = float_to_bits(b) >> 16
    if a_bits & 0x8000:
        a_bits = 0x8000 - (a_bits & 0x7FFF)
    if b_bits & 0x8000:
        b_bits = 0x8000 - (b_bits & 0x7FFF)
    return abs(a_bits - b_bits)


def check_ulp(actual, expected, max_ulp, dtype_name):
    """Check ULP errors across all elements. Returns (max_ulp_found, num_violations)."""
    actual_flat = actual.flatten().tolist()
    expected_flat = expected.flatten().tolist()
    max_ulp_found = 0
    violations = []
    ulp_fn = ulp_difference_bf16 if dtype_name == "bfloat16" else ulp_difference_fp32

    for i, (a, e) in enumerate(zip(actual_flat, expected_flat)):
        ulp = ulp_fn(a, e)
        if ulp > max_ulp_found:
            max_ulp_found = ulp
        if ulp > max_ulp:
            violations.append((i, a, e, ulp))

    return max_ulp_found, violations


@pytest.fixture(scope="module")
def device():
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


@pytest.mark.parametrize("cap", [50.0, 10.0, 1.0])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_softcap_basic(device, cap, dtype):
    """Test softcap with varied cap values and dtypes."""
    torch.manual_seed(42)
    shape = [1, 1, 32, 32]

    # Generate inputs spanning different tanh regimes
    x = torch.randn(shape, dtype=torch.float32) * cap * 2

    expected = torch_softcap(x, cap)

    tt_input = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.softcap(tt_input, cap=cap)
    actual = ttnn.to_torch(tt_output).to(torch.float32)

    if dtype == ttnn.bfloat16:
        max_ulp, violations = check_ulp(actual, expected, max_ulp=2, dtype_name="bfloat16")
        assert len(violations) == 0, (
            f"bfloat16 ULP violations (cap={cap}): {len(violations)}/{actual.numel()}, "
            f"max_ulp={max_ulp}, first 5: {violations[:5]}"
        )
    else:
        # [3,3] Padé rational approximation has ~0.05% relative error for |u| < 3
        # and clamps to ±1 beyond. Use relative tolerance for fp32.
        assert torch.allclose(actual, expected, rtol=5e-3, atol=5e-2), (
            f"fp32 allclose failed (cap={cap}): max_abs_err={torch.max(torch.abs(actual - expected)).item():.6f}, "
            f"max_rel_err={torch.max(torch.abs((actual - expected) / (expected + 1e-10))).item():.6f}"
        )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_softcap_edge_cases(device, dtype):
    """Test softcap with edge case inputs: zero, very small, very large."""
    cap = 50.0
    # Include: 0, small positive, small negative, large positive, large negative,
    # exactly cap, -cap, 2*cap (should saturate near cap)
    edge_values = [0.0, 1e-6, -1e-6, 0.25, -0.25, 1.0, -1.0, cap * 0.5, -cap * 0.5, cap, -cap, cap * 5, -cap * 5]
    # Pad to fill a 32x32 tile
    while len(edge_values) < 1024:
        edge_values.append(edge_values[len(edge_values) % len(edge_values[:13])])
    x = torch.tensor(edge_values[:1024], dtype=torch.float32).reshape(1, 1, 32, 32)

    expected = torch_softcap(x, cap)

    tt_input = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.softcap(tt_input, cap=cap)
    actual = ttnn.to_torch(tt_output).to(torch.float32)

    if dtype == ttnn.bfloat16:
        max_ulp, violations = check_ulp(actual, expected, max_ulp=2, dtype_name="bfloat16")
        assert len(violations) == 0, (
            f"bfloat16 edge case ULP violations: {len(violations)}, " f"max_ulp={max_ulp}, first 5: {violations[:5]}"
        )
    else:
        assert torch.allclose(
            actual, expected, rtol=5e-3, atol=5e-2
        ), f"fp32 edge case allclose failed: max_abs_err={torch.max(torch.abs(actual - expected)).item():.6f}"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_softcap_multi_tile(device, dtype):
    """Test softcap with multiple tiles to verify tile iteration."""
    cap = 30.0
    torch.manual_seed(123)
    shape = [1, 1, 64, 64]  # 4 tiles
    x = torch.randn(shape, dtype=torch.float32) * cap * 3

    expected = torch_softcap(x, cap)

    tt_input = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.softcap(tt_input, cap=cap)
    actual = ttnn.to_torch(tt_output).to(torch.float32)

    if dtype == ttnn.bfloat16:
        max_ulp, violations = check_ulp(actual, expected, max_ulp=2, dtype_name="bfloat16")
        assert len(violations) == 0, (
            f"bfloat16 multi-tile ULP violations: {len(violations)}/{actual.numel()}, " f"max_ulp={max_ulp}"
        )
    else:
        assert torch.allclose(actual, expected, rtol=5e-3, atol=5e-2), (
            f"fp32 multi-tile allclose failed: " f"max_abs_err={torch.max(torch.abs(actual - expected)).item():.6f}"
        )

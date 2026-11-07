# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import pytest
from tests.ttnn.utils_for_testing import assert_with_ulp

pytestmark = pytest.mark.use_module_device


def create_full_range_tensor(input_shape, dtype, value_ranges):
    num_elements = torch.prod(torch.tensor(input_shape)).item()
    num_ranges = len(value_ranges)
    elements_per_range = num_elements // num_ranges
    leftover = num_elements % num_ranges

    segments = []
    for i, (low, high) in enumerate(value_ranges):
        n = elements_per_range + (1 if i < leftover else 0)
        segments.append(torch.linspace(low, high, steps=n, dtype=dtype))
    return torch.cat(segments).reshape(input_shape)


shape = torch.Size([1, 2, 32, 128])

DIV_RANGES_A = [
    (-100, 100),
    (-300, 300),
    (-750, 500),
    (-1000, 1000),
    (-1e4, 1e4),
    (-1e5, 1e5),
    (-1e7, 1e7),
    (-1e10, 1e10),
    (-1e15, 1e15),
    (1e8, 1e10),
    (1e12, 1e15),
    (-1e10, -1e8),
    (-1e15, -1e12),
    (-1e-5, 1e-5),
    (-1e-10, 1e-10),
]
DIV_RANGES_B = [
    (-50, 200),
    (-400, 600),
    (-2000, 3000),
    (-2e4, 2e4),
    (-3e5, 3e5),
    (-5e6, 5e6),
    (-2e8, 2e8),
    (-8e9, 8e9),
    (-1e14, 1e14),
    (2e8, 5e9),
    (8e11, 8e14),
    (-5e8, -2e7),
    (-8e14, -8e11),
    (-2e-4, 2e-4),
    (-2e-8, 2e-8),
]

MOD_RANGES_A = [
    (-100, 100),
    (-300, 300),
    (-1000, 1000),
    (-1e4, 1e4),
    (-5e4, 5e4),
    (0, 100),
    (-100, 0),
    (-1, 1),
    (-1e-2, 1e-2),
    (-1e-4, 1e-4),
    (50, 5000),
    (-5000, -50),
    (-2e4, 2e4),
    (1, 1e3),
    (-1e3, -1),
]
MOD_RANGES_B = [
    (0.5, 50),
    (1, 200),
    (2, 500),
    (5, 2000),
    (10, 5000),
    (0.5, 10),
    (-50, -0.5),
    (0.25, 4),
    (-4, -0.25),
    (0.01, 2),
    (20, 800),
    (-800, -20),
    (3, 3000),
    (0.5, 100),
    (-100, -0.5),
]


def test_remainder_fp32(device):
    torch_input_tensor_a = create_full_range_tensor(shape, torch.float32, MOD_RANGES_A)
    torch_input_tensor_b = create_full_range_tensor(shape, torch.float32, MOD_RANGES_B)

    golden_fn = ttnn.get_golden_function(ttnn.remainder)
    torch_output = golden_fn(torch_input_tensor_a, torch_input_tensor_b, device=device)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output = ttnn.remainder(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert torch.allclose(output, torch_output, atol=1e-2, rtol=1e-3, equal_nan=True)


def test_fmod_fp32(device):
    torch_input_tensor_a = create_full_range_tensor(shape, torch.float32, MOD_RANGES_A)
    torch_input_tensor_b = create_full_range_tensor(shape, torch.float32, MOD_RANGES_B)

    golden_fn = ttnn.get_golden_function(ttnn.fmod)
    torch_output = golden_fn(torch_input_tensor_a, torch_input_tensor_b, device=device)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output = ttnn.fmod(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert torch.allclose(output, torch_output, atol=1e-2, rtol=1e-3, equal_nan=True)


def test_div_no_nan_fp32(device):
    torch.manual_seed(0)
    torch_input_tensor_a = create_full_range_tensor(shape, torch.float32, DIV_RANGES_A)
    torch_input_tensor_b = create_full_range_tensor(shape, torch.float32, DIV_RANGES_B)
    # No-nan behavior
    b_flat = torch_input_tensor_b.flatten()
    b_flat[::101] = 0.0
    torch_input_tensor_b = b_flat.reshape(shape)

    golden_fn = ttnn.get_golden_function(ttnn.div_no_nan)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    torch_output = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    output = ttnn.div_no_nan(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)
    assert_with_ulp(expected_result=torch_output, actual_result=output, ulp_threshold=1, allow_nonfinite=True)


@pytest.mark.parametrize("op", [ttnn.div_no_nan, ttnn.divide])
def test_div_fp32_quotient_refinement(device, op):
    # Scale both adversarial pairs across the normal exponent range. Refining
    # unscaled operands loses a subnormal residual even when a, b and a/b are normal.
    a = torch.tensor([0x4F518358, 0x3FCF913C], dtype=torch.int32).view(torch.float32).double()
    b = torch.tensor([0x4EDD39E7, 0x3FE28788], dtype=torch.int32).view(torch.float32).double()
    shifts = torch.arange(-126, 127)[:, None] - torch.tensor([31, 0])
    a, b = torch.ldexp(a, shifts).float().flatten(), torch.ldexp(b, shifts).float().flatten()
    normal = (a >= torch.finfo(torch.float32).tiny) & (b >= torch.finfo(torch.float32).tiny)
    a, b = a[normal], b[normal]
    a, b = torch.cat([a, -a, a, -a]), torch.cat([b, b, -b, -b])
    a, b = a.repeat(2)[:2048].reshape(64, 32), b.repeat(2)[:2048].reshape(64, 32)
    input_a = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    input_b = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(op(input_a, input_b))
    expected = (a.double() / b.double()).float()
    assert_with_ulp(expected_result=expected, actual_result=actual, ulp_threshold=1)


@pytest.mark.parametrize("op", [ttnn.div_no_nan, ttnn.divide])
def test_div_fp32_random_exponents(device, op):
    generator = torch.Generator().manual_seed(58228)
    a = torch.randint(0x00800000, 0x7F800000, (65536,), generator=generator, dtype=torch.int32).view(torch.float32)
    b = torch.randint(0x00800000, 0x7F800000, (65536,), generator=generator, dtype=torch.int32).view(torch.float32)
    a[::2], b[::3] = -a[::2], -b[::3]
    expected = (a.double() / b.double()).float()
    normal = torch.isfinite(expected) & (expected.abs() >= torch.finfo(torch.float32).tiny)
    # Keep a fixed, tile-aligned shape without making accuracy claims about FTZ results.
    a, b = torch.where(normal, a, 1.0).reshape(2048, 32), torch.where(normal, b, 1.0).reshape(2048, 32)
    expected = (a.double() / b.double()).float()
    input_a = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    input_b = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(op(input_a, input_b))
    assert_with_ulp(expected_result=expected, actual_result=actual, ulp_threshold=1)


@pytest.mark.parametrize("op", [ttnn.div_no_nan, ttnn.divide])
def test_div_fp32_mantissa_refinement(device, op):
    # Sweep near the reciprocal seed's worst error, varying both mantissas.
    # With one reciprocal Newton step, a separately rounded residual multiply
    # on Wormhole fails this at 2 ULP.
    bits = torch.arange(1 << 18, dtype=torch.int32)
    a = (((bits * 1664525 + 1013904223) & 0x7FFFFF) | 0x3F800000).view(torch.float32).reshape(8192, 32)
    b = (bits | 0x3F800000).view(torch.float32).reshape(8192, 32)
    expected = (a.double() / b.double()).float()
    input_a = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    input_b = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(op(input_a, input_b))
    assert_with_ulp(expected_result=expected, actual_result=actual, ulp_threshold=1)


@pytest.mark.parametrize("op", [ttnn.div_no_nan, ttnn.divide])
def test_div_fp32_exponent_boundaries(device, op):
    limits = torch.finfo(torch.float32)
    values = torch.tensor([limits.tiny, 2.0 * limits.tiny, 0.5, 1.0, 2.0, limits.max / 2, limits.max, 2.0**64])
    a, b = torch.meshgrid(values, values, indexing="ij")
    a, b = a.flatten().repeat(16).reshape(32, 32), b.flatten().repeat(16).reshape(32, 32)
    a[::2], b[::3] = -a[::2], -b[::3]
    expected = (a.double() / b.double()).float()
    # Subnormal results flush to zero on these architectures.
    expected = torch.where(expected.abs() < limits.tiny, 0.0, expected)
    input_a = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    input_b = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(op(input_a, input_b))
    assert_with_ulp(expected_result=expected, actual_result=actual, ulp_threshold=1, allow_nonfinite=True)


@pytest.mark.parametrize("op", [ttnn.div_no_nan, ttnn.divide])
def test_div_fp32_special_values(device, op):
    tiny = torch.finfo(torch.float32).tiny
    values = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, tiny, -tiny, tiny / 2, -tiny / 2, 2.0**-149, -(2.0**-149)]
        + [float("inf"), -float("inf"), float("nan"), -float("nan")]
    )
    a, b = torch.meshgrid(values, values, indexing="ij")
    a, b = a.flatten().repeat(4).reshape(32, 32), b.flatten().repeat(4).reshape(32, 32)
    # Subnormal inputs flush to sign-preserving zero. No normal pair here has a subnormal
    # quotient, whose zero sign differs by architecture; exponent_boundaries covers those.
    flush = lambda x: torch.where(x.abs() < tiny, torch.copysign(torch.zeros_like(x), x), x)
    expected = flush(a) / flush(b)
    if op == ttnn.div_no_nan:
        # eqz compares exactly, so subnormal divisors keep division's flushed result.
        expected = torch.where(b == 0, 0.0, expected)
    # SFPU multiplication produces positive zero for exact zero results.
    expected = torch.where(expected == 0, 0.0, expected)
    input_a = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    input_b = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(op(input_a, input_b))
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    non_nan = ~torch.isnan(expected)
    assert torch.equal(actual[non_nan], expected[non_nan])
    assert torch.equal(torch.signbit(actual[non_nan]), torch.signbit(expected[non_nan]))


@pytest.mark.parametrize("val_a, val_b", [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.0)])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("approx", [True, False])
def test_div_by_zero(device, val_a, val_b, dtype, approx):
    torch_dtype = getattr(torch, dtype)
    tt_dtype = getattr(ttnn, dtype)

    if val_a == 0.0 and val_b == 0.0 and dtype == "bfloat16":
        pytest.skip("Skipping test for 0/0 on bfloat16")

    x_torch = torch.tensor([[val_a]], dtype=torch_dtype)
    y_torch = torch.tensor([[val_b]], dtype=torch_dtype)

    golden_fn = ttnn.get_golden_function(ttnn.divide)
    z_torch = golden_fn(x_torch, y_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    z_tt_div = ttnn.divide(x_tt, y_tt, fast_and_approximate_mode=approx)
    tt_out = ttnn.to_torch(z_tt_div)

    # Note: torch.equal return false for if both tensors are nan
    # This is why we use assert_with_ulp to test for equality

    if approx and dtype == "bfloat16":
        pytest.skip("Skipping test for fast approximate mode")

    assert_with_ulp(expected_result=z_torch, actual_result=tt_out, ulp_threshold=0, allow_nonfinite=True)


@pytest.mark.parametrize("val_a, val_b", [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.0)])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("approx", [True, False])
def test_divide_inplace_by_zero(device, val_a, val_b, dtype, approx):
    torch_dtype = getattr(torch, dtype)
    tt_dtype = getattr(ttnn, dtype)

    if val_a == 0.0 and val_b == 0.0 and dtype == "bfloat16":
        pytest.skip("Skipping test for 0/0 on bfloat16")

    x_torch = torch.tensor([[val_a]], dtype=torch_dtype)
    y_torch = torch.tensor([[val_b]], dtype=torch_dtype)

    golden_fn = ttnn.get_golden_function(ttnn.divide)
    z_torch = golden_fn(x_torch, y_torch)

    x_tt_inplace = ttnn.from_torch(x_torch, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt_inplace = ttnn.from_torch(y_torch, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.divide_(x_tt_inplace, y_tt_inplace, fast_and_approximate_mode=approx)
    tt_out_inplace = ttnn.to_torch(x_tt_inplace)

    if approx and dtype == "bfloat16":
        pytest.skip("Skipping test for fast approximate mode")

    assert_with_ulp(expected_result=z_torch, actual_result=tt_out_inplace, ulp_threshold=0, allow_nonfinite=True)


@pytest.mark.parametrize(
    "testing_dtype",
    ["bfloat16", "float32"],
)
def test_remainder_nan(testing_dtype, device):
    torch_dtype = getattr(torch, testing_dtype)
    ttnn_dtype = getattr(ttnn, testing_dtype)
    if testing_dtype == "bfloat16":
        pytest.xfail("NaN is packed as inf for ttnn.bfloat16")

    torch_input_a = torch.tensor([1.0, 0.0, -1.0], dtype=torch_dtype)
    torch_input_b = torch.tensor([0.0, 0.0, 0.0], dtype=torch_dtype)

    golden_function = ttnn.get_golden_function(ttnn.remainder)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    tt_in_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.remainder(tt_in_a, tt_in_b)
    output_tensor = ttnn.to_torch(tt_result)

    assert torch.equal(torch.isnan(golden), torch.isnan(output_tensor))


@pytest.mark.parametrize(
    "testing_dtype",
    ["bfloat16", "float32"],
)
def test_fmod_nan(testing_dtype, device):
    torch_dtype = getattr(torch, testing_dtype)
    ttnn_dtype = getattr(ttnn, testing_dtype)
    if testing_dtype == "bfloat16":
        pytest.xfail("NaN is packed as inf for ttnn.bfloat16")

    torch_input_a = torch.tensor([1.0, 0.0, -1.0], dtype=torch_dtype)
    torch_input_b = torch.tensor([0.0, 0.0, 0.0], dtype=torch_dtype)

    golden_function = ttnn.get_golden_function(ttnn.fmod)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    tt_in_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.fmod(tt_in_a, tt_in_b)
    output_tensor = ttnn.to_torch(tt_result)

    assert torch.equal(torch.isnan(golden), torch.isnan(output_tensor))


_QUASAR_MODULO_SKIP = "The Quasar remainder and fmod kernels have not received these float fixes yet"


def _skip_on_quasar(device):
    if device.arch() == ttnn.device.Arch.QUASAR:
        pytest.skip(_QUASAR_MODULO_SKIP)


def _opposite_sign_exact_multiples(torch_dtype, count, seed=0):
    """Pairs (a, b) with a = n * |b| exactly representable in torch_dtype and
    sign(a) != sign(b), so remainder(a, b) is exactly 0. Deterministic (seeded).
    n is drawn first and b gets at most (precision - n.bit_length()) significant
    bits, so n * b always fits the dtype's significand and every draw is exact."""
    precision, max_n = (24, 4096) if torch_dtype == torch.float32 else (8, 64)
    gen = torch.Generator().manual_seed(seed)
    n = torch.randint(2, max_n, (count,), generator=gen)
    exponent = torch.randint(-3, 18, (count,), generator=gen)
    a_vals, b_vals = [], []
    for i in range(count):
        b_bits = precision - int(n[i]).bit_length()
        significand = int(torch.randint(2 ** (b_bits - 1), 2**b_bits, (1,), generator=gen).item())
        b = significand * 2.0 ** (int(exponent[i]) - b_bits + 1)
        a = int(n[i]) * b
        if i % 2 == 0:
            a_vals.append(a)
            b_vals.append(-b)
        else:
            a_vals.append(-a)
            b_vals.append(b)
    a_tensor = torch.tensor(a_vals, dtype=torch_dtype)
    b_tensor = torch.tensor(b_vals, dtype=torch_dtype)
    assert torch.equal(a_tensor.double(), torch.tensor(a_vals, dtype=torch.float64))
    assert torch.equal(b_tensor.double(), torch.tensor(b_vals, dtype=torch.float64))
    return a_tensor, b_tensor


# remainder(a, b) must be exactly 0 when a is an exact multiple of b. The kernel computes
# a - floor(a * (1/b)) * b with an approximate reciprocal; when the quotient lands just below the
# true integer the floor steps one past it and the raw result is b itself (or, at large magnitudes,
# b plus the rounding of that product). The kernel repairs that overshoot for same-sign operands but
# used to skip the repair when a and b differ in sign, so e.g. remainder(91.0, -13.0) came back as
# -13.0 instead of 0.0.
@pytest.mark.parametrize(
    "testing_dtype, pairs",
    [
        (
            "float32",
            [
                (91.0, -13.0),
                (-91.0, 13.0),
                (829680.6875, -462.991455078125),
                (424466.4375, -421.09765625),
                (20978266.0, -42988.25),
                (-1234226.25, 3214.130859375),
                (14056288.0, -5818.0),
                (63853748.0, -149191.0),
            ],
        ),
        ("bfloat16", [(-1512.0, 63.0), (-390.0, 13.0), (41.25, -1.875), (675840.0, -3840.0), (2.328125, -1.1640625)]),
    ],
)
def test_remainder_opposite_sign_exact_multiple(testing_dtype, pairs, device):
    _skip_on_quasar(device)
    torch_dtype = getattr(torch, testing_dtype)
    ttnn_dtype = getattr(ttnn, testing_dtype)

    explicit_a = torch.tensor([a for a, _ in pairs], dtype=torch_dtype)
    explicit_b = torch.tensor([b for _, b in pairs], dtype=torch_dtype)
    random_a, random_b = _opposite_sign_exact_multiples(torch_dtype, 1024 - len(pairs))
    torch_input_a = torch.cat([explicit_a, random_a]).reshape(32, 32)
    torch_input_b = torch.cat([explicit_b, random_b]).reshape(32, 32)
    # Every pair is an exact multiple in this dtype, so the whole golden is 0.
    assert torch.all(torch_input_a.double() % torch_input_b.double() == 0)

    golden_function = ttnn.get_golden_function(ttnn.remainder)
    golden = golden_function(torch_input_a, torch_input_b, device=device)
    assert torch.all(golden == 0)

    tt_in_a = ttnn.from_torch(torch_input_a, dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT)
    tt_in_b = ttnn.from_torch(torch_input_b, dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.to_torch(ttnn.remainder(tt_in_a, tt_in_b))

    mismatch = output_tensor != golden
    assert not mismatch.any(), (
        f"{int(mismatch.sum())} of {mismatch.numel()} exact multiples gave a non-zero remainder; first: "
        f"remainder({torch_input_a[mismatch][0].item()}, {torch_input_b[mismatch][0].item()}) = "
        f"{output_tensor[mismatch][0].item()}"
    )


# Quotients of 2^24 and above: floor(a / b) can no longer be stepped by one in float, so the
# overshoot correction has to fall back to subtracting b from the raw result. The non-multiple
# pairs are the ones the kernel resolves at this magnitude on Wormhole, where the reciprocal of 25
# rounds down so a / b lands just below the true quotient, leaving |result| >= |b| once;
# remainder(419430432.0, 25.0) returned 32.0 instead of 7.0 without the fallback. Blackhole seeds
# its reciprocal differently and was not run locally; if its reciprocal of 25 rounds up these two
# pairs land one above the quotient instead and Blackhole CI decides whether they hold there.
@pytest.mark.parametrize(
    "testing_dtype, pairs",
    [
        (
            "float32",
            [
                (419430432.0, 25.0),
                (-419430432.0, -25.0),
                (419430400.0, 25.0),
                (419430400.0, -25.0),
                (-419430400.0, 25.0),
                (-838860800.0, -25.0),
                (50331648.0, -3.0),
                (-50331648.0, 3.0),
                (125829120.0, 7.5),
                (-251658240.0, 7.5),
            ],
        ),
        (
            "bfloat16",
            [
                (419430400.0, 25.0),
                (419430400.0, -25.0),
                (-524288000.0, 25.0),
                (-838860800.0, -25.0),
                (402653184.0, -3.0),
                (-402653184.0, 24.0),
            ],
        ),
    ],
)
def test_remainder_large_quotient(testing_dtype, pairs, device):
    _skip_on_quasar(device)
    torch_dtype = getattr(torch, testing_dtype)
    ttnn_dtype = getattr(ttnn, testing_dtype)

    torch_input_a = torch.tensor([a for a, _ in pairs], dtype=torch_dtype)
    torch_input_b = torch.tensor([b for _, b in pairs], dtype=torch_dtype)
    assert torch.all(torch_input_a.abs() / torch_input_b.abs() >= 2.0**24)

    golden_function = ttnn.get_golden_function(ttnn.remainder)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    tt_in_a = ttnn.from_torch(torch_input_a, dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT)
    tt_in_b = ttnn.from_torch(torch_input_b, dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.to_torch(ttnn.remainder(tt_in_a, tt_in_b))

    assert torch.equal(output_tensor, golden), f"device {output_tensor.tolist()} vs torch {golden.tolist()}"


# Small quotients whose correctly rounded remainder equals the divisor: a = -1, b = 2^26 gives
# floor(a / b) = -1 and a - (-1) * b = 2^26 - 1, which rounds to b itself in float32 (torch returns
# b too). The remainder must be left alone: a |result| >= |b| repair applied here returned 0. The
# last pair is an exact multiple of a divisor whose approximate reciprocal can be far off (2^126 on
# Blackhole, where the hardware seed returns 0 for |b| >= 2^126); the floor then lands two above
# the true quotient and a single sign correction leaves -b. Blackhole is not run locally.
@pytest.mark.parametrize("testing_dtype", ["bfloat16", "float32"])
def test_remainder_result_equal_to_divisor(testing_dtype, device):
    _skip_on_quasar(device)
    torch_dtype = getattr(torch, testing_dtype)
    ttnn_dtype = getattr(ttnn, testing_dtype)

    torch_input_a = torch.tensor([-1.0, 1.0, -(2.0**-25), 2.0**-25, -(2.0**127), 2.0**127], dtype=torch_dtype)
    torch_input_b = torch.tensor([2.0**26, -(2.0**26), 1.0, -1.0, 2.0**126, -(2.0**126)], dtype=torch_dtype)

    golden_function = ttnn.get_golden_function(ttnn.remainder)
    golden = golden_function(torch_input_a, torch_input_b, device=device)
    assert torch.equal(golden, torch.tensor([2.0**26, -(2.0**26), 1.0, -1.0, 0.0, 0.0], dtype=torch_dtype))

    tt_in_a = ttnn.from_torch(torch_input_a, dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT)
    tt_in_b = ttnn.from_torch(torch_input_b, dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.to_torch(ttnn.remainder(tt_in_a, tt_in_b))

    assert torch.equal(output_tensor, golden), f"device {output_tensor.tolist()} vs torch {golden.tolist()}"


_INF = float("inf")


def _negative_nan(torch_dtype):
    """A quiet NaN with the sign bit set (0xFFC00000 / 0xFFC0)."""
    if torch_dtype == torch.bfloat16:
        return torch.tensor([-0x40], dtype=torch.int16).view(torch.bfloat16)
    return torch.tensor([-0x400000], dtype=torch.int32).view(torch.float32)


def _assert_nan_lanes(testing_dtype, output_lanes):
    if testing_dtype == "bfloat16":
        # NaN is packed as inf for ttnn.bfloat16; either way the lane must not be a finite number.
        assert not torch.isfinite(output_lanes).any(), f"device {output_lanes.tolist()}"
    else:
        assert torch.isnan(output_lanes).all(), f"device {output_lanes.tolist()}"


# remainder(a, +-inf) for finite a is a when a is zero or has the divisor's sign, else the
# divisor itself (torch semantics). The kernel's floor(a * (1/b)) * b term is 0 * inf for an
# infinite divisor, which used to leave NaN in every such lane.
@pytest.mark.parametrize("testing_dtype", ["bfloat16", "float32"])
def test_remainder_inf_divisor(testing_dtype, device):
    _skip_on_quasar(device)
    torch_dtype = getattr(torch, testing_dtype)
    ttnn_dtype = getattr(ttnn, testing_dtype)

    torch_input_a = torch.tensor(
        [5.0, -5.0, 5.0, -5.0, 0.0, 0.0, 2.5, -2.5, _INF, -_INF, _INF, -_INF, 0.0, 0.0], dtype=torch_dtype
    )
    torch_input_b = torch.tensor(
        [_INF, _INF, -_INF, -_INF, _INF, -_INF, _INF, -_INF, _INF, _INF, -_INF, -_INF, _INF, -_INF],
        dtype=torch_dtype,
    )
    # A NaN dividend with the sign bit set (0xFFC00000): SFPABS keeps that sign, so a magnitude
    # test built on it would classify the lane as finite and negative, and return the divisor.
    torch_input_a[12:] = _negative_nan(torch_dtype)

    golden_function = ttnn.get_golden_function(ttnn.remainder)
    golden = golden_function(torch_input_a, torch_input_b, device=device)
    # An infinite or NaN dividend stays NaN whatever the divisor's sign.
    nan_lanes = torch.isnan(golden)
    assert torch.equal(nan_lanes, ~torch.isfinite(torch_input_a))

    tt_in_a = ttnn.from_torch(torch_input_a, dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT)
    tt_in_b = ttnn.from_torch(torch_input_b, dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.to_torch(ttnn.remainder(tt_in_a, tt_in_b))

    assert torch.equal(
        output_tensor[~nan_lanes], golden[~nan_lanes]
    ), f"device {output_tensor.tolist()} vs torch {golden.tolist()}"
    _assert_nan_lanes(testing_dtype, output_tensor[nan_lanes])


# fmod(a, +-inf) is a for finite a and NaN for an infinite a (torch semantics). The kernel's
# trunc(a * (1/b)) * b term is 0 * inf for an infinite divisor, which used to leave NaN in the
# finite-a lanes, while the a == b special case zeroed inf/inf.
@pytest.mark.parametrize("testing_dtype", ["bfloat16", "float32"])
def test_fmod_inf_divisor(testing_dtype, device):
    _skip_on_quasar(device)
    torch_dtype = getattr(torch, testing_dtype)
    ttnn_dtype = getattr(ttnn, testing_dtype)

    torch_input_a = torch.tensor(
        [5.0, -5.0, 5.0, -5.0, 0.0, 2.5, _INF, -_INF, _INF, -_INF, 0.0, 0.0], dtype=torch_dtype
    )
    torch_input_b = torch.tensor(
        [_INF, _INF, -_INF, -_INF, _INF, -_INF, _INF, -_INF, -_INF, _INF, _INF, -_INF], dtype=torch_dtype
    )
    # A NaN dividend with the sign bit set (0xFFC00000), which SFPABS leaves sign-set.
    torch_input_a[10:] = _negative_nan(torch_dtype)

    golden_function = ttnn.get_golden_function(ttnn.fmod)
    golden = golden_function(torch_input_a, torch_input_b, device=device)
    finite_a = torch.isfinite(torch_input_a)
    assert torch.equal(torch.isnan(golden), ~finite_a)

    tt_in_a = ttnn.from_torch(torch_input_a, dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT)
    tt_in_b = ttnn.from_torch(torch_input_b, dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.to_torch(ttnn.fmod(tt_in_a, tt_in_b))

    assert torch.equal(
        output_tensor[finite_a], golden[finite_a]
    ), f"device {output_tensor.tolist()} vs torch {golden.tolist()}"
    _assert_nan_lanes(testing_dtype, output_tensor[~finite_a])


# The scalar-divisor forms route to the unary kernels (ckernel_sfpu_remainder.h / ckernel_sfpu_fmod.h),
# which receive a host reciprocal of 0 for an infinite divisor and used to return NaN where the
# tensor-tensor path returns the torch value.
@pytest.mark.parametrize("ttnn_op", [ttnn.remainder, ttnn.fmod])
@pytest.mark.parametrize("scalar", [_INF, -_INF])
@pytest.mark.parametrize("testing_dtype", ["bfloat16", "float32"])
def test_modulo_inf_scalar_divisor(ttnn_op, scalar, testing_dtype, device):
    _skip_on_quasar(device)
    torch_dtype = getattr(torch, testing_dtype)
    ttnn_dtype = getattr(ttnn, testing_dtype)

    torch_input_a = torch.tensor([5.0, -5.0, 0.0, 2.5, -2.5, _INF, -_INF, 0.0], dtype=torch_dtype)
    torch_input_a[7:] = _negative_nan(torch_dtype)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(torch_input_a, scalar, device=device)
    nan_lanes = torch.isnan(golden)
    assert torch.equal(nan_lanes, ~torch.isfinite(torch_input_a))

    tt_in_a = ttnn.from_torch(torch_input_a, dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.to_torch(ttnn_op(tt_in_a, scalar))

    assert torch.equal(
        output_tensor[~nan_lanes], golden[~nan_lanes]
    ), f"device {output_tensor.tolist()} vs torch {golden.tolist()}"
    _assert_nan_lanes(testing_dtype, output_tensor[nan_lanes])


def test_optional_output_tensor_remainder(device):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.tensor([[5.0, 7.0, -5.0, -7.0, 3.5, 10.0, 1.5, -1.5, 9.0, 15.0]], dtype=torch.bfloat16)
    torch_input_tensor_b = torch.tensor([[2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 0.5, 0.5, -2.0, -4.0]], dtype=torch.bfloat16)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    golden_function = ttnn.get_golden_function(ttnn.remainder)
    torch_golden = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    torch_opt_output_tensor = torch.zeros_like(torch_golden)
    optional_output_tensor = ttnn.from_torch(torch_opt_output_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn.remainder(
        input_tensor_a,
        input_tensor_b,
        output_tensor=optional_output_tensor,
    )
    optional_output_tensor = ttnn.to_torch(optional_output_tensor)

    assert_with_ulp(expected_result=torch_golden, actual_result=optional_output_tensor, ulp_threshold=0)


@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("rounding_mode", [None, "floor", "trunc"])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32])
@pytest.mark.parametrize("op", [ttnn.div, ttnn.divide])
def test_div_output_dtype(device, input_dtype, rounding_mode, output_dtype, op):
    torch_dtype = torch.bfloat16 if input_dtype == ttnn.bfloat16 else torch.float32
    torch_input_a = torch.tensor([[7.0, -7.0, 6.0, -6.0]], dtype=torch_dtype)
    torch_input_b = torch.full_like(torch_input_a, 2.0)

    input_tensor_a = ttnn.from_torch(torch_input_a, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_b, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = op(input_tensor_a, input_tensor_b, rounding_mode=rounding_mode, dtype=output_dtype)

    torch_golden = torch.div(torch_input_a.float(), torch_input_b.float(), rounding_mode=rounding_mode)
    if output_dtype == ttnn.int32:
        torch_golden = torch_golden.to(torch.int32)

    assert output.dtype == output_dtype
    assert torch.equal(ttnn.to_torch(output).float(), torch_golden.float())


@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
def test_div_int32_output_rounds_before_narrowing(device, rounding_mode):
    # Narrowing the quotient before rounding would truncate toward zero and give floor() the wrong
    # answer for negatives, so the whole negative range has to agree with torch.
    torch_input_a = torch.arange(-64, 64, dtype=torch.float32).reshape(1, 1, 8, 16)
    torch_input_b = torch.full_like(torch_input_a, 3.0)

    input_tensor_a = ttnn.from_torch(torch_input_a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.div(input_tensor_a, input_tensor_b, rounding_mode=rounding_mode, dtype=ttnn.int32)
    torch_golden = torch.div(torch_input_a, torch_input_b, rounding_mode=rounding_mode).to(torch.int32)

    assert output.dtype == ttnn.int32
    assert torch.equal(ttnn.to_torch(output), torch_golden)


@pytest.mark.parametrize("rounding_mode", [None, "floor", "trunc"])
@pytest.mark.parametrize("output_dtype", [ttnn.float32, ttnn.int32])
def test_div_int32_inputs_output_dtype(device, rounding_mode, output_dtype):
    if rounding_mode is None and output_dtype != ttnn.float32:
        pytest.skip("Integer division with rounding_mode=None only supports a float32 output")

    torch_input_a = torch.tensor([[7, -7, 6, -6]], dtype=torch.int32)
    torch_input_b = torch.full_like(torch_input_a, 2)

    input_tensor_a = ttnn.from_torch(torch_input_a, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_b, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.div(input_tensor_a, input_tensor_b, rounding_mode=rounding_mode, dtype=output_dtype)
    torch_golden = torch.div(torch_input_a, torch_input_b, rounding_mode=rounding_mode).float()

    assert output.dtype == output_dtype
    assert torch.equal(ttnn.to_torch(output).float(), torch_golden)


@pytest.mark.parametrize("rounding_mode", [None, "floor", "trunc"])
@pytest.mark.parametrize("output_dtype", [ttnn.float32, ttnn.int32])
def test_div_scalar_output_dtype(device, rounding_mode, output_dtype):
    torch_input = torch.tensor([[7.0, -7.0, 6.0, -6.0]], dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.div(input_tensor, 2.0, rounding_mode=rounding_mode, dtype=output_dtype)

    torch_golden = torch.div(torch_input.float(), 2.0, rounding_mode=rounding_mode)
    if output_dtype == ttnn.int32:
        torch_golden = torch_golden.to(torch.int32)

    assert output.dtype == output_dtype
    assert torch.equal(ttnn.to_torch(output).float(), torch_golden.float())


@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
@pytest.mark.parametrize("output_dtype", [ttnn.int32, ttnn.float32])
def test_div_preallocated_output_rounds_before_narrowing(device, rounding_mode, output_dtype):
    # A preallocated integer output has to reach the same value as dtype=int32, so the quotient
    # cannot be narrowed into it before floor()/trunc() runs.
    torch_input_a = torch.tensor([[7.0, -7.0, 6.0, -6.0]], dtype=torch.bfloat16)
    torch_input_b = torch.full_like(torch_input_a, 2.0)

    input_tensor_a = ttnn.from_torch(torch_input_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    torch_dtype = torch.int32 if output_dtype == ttnn.int32 else torch.float32
    preallocated = ttnn.from_torch(
        torch.zeros_like(torch_input_a, dtype=torch_dtype), dtype=output_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )

    output = ttnn.div(input_tensor_a, input_tensor_b, rounding_mode=rounding_mode, output_tensor=preallocated)

    torch_golden = torch.div(torch_input_a.float(), torch_input_b.float(), rounding_mode=rounding_mode)

    assert output.dtype == output_dtype
    assert torch.equal(ttnn.to_torch(output).float(), torch_golden)
    # the caller's tensor must hold the result, not just the returned handle
    assert torch.equal(ttnn.to_torch(preallocated).float(), torch_golden)


@pytest.mark.parametrize("rounding_mode", [None, "floor", "trunc"])
@pytest.mark.parametrize("output_dtype", [ttnn.int32, ttnn.float32])
def test_div_output_dtype_with_sub_core_grids(device, rounding_mode, output_dtype):
    # The post-rounding typecast has to keep the caller's core restriction.
    torch_input_a = torch.tensor([[7.0, -7.0, 6.0, -6.0]], dtype=torch.bfloat16)
    torch_input_b = torch.full_like(torch_input_a, 2.0)

    input_tensor_a = ttnn.from_torch(torch_input_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sub_core_grids = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))])

    output = ttnn.div(
        input_tensor_a,
        input_tensor_b,
        rounding_mode=rounding_mode,
        dtype=output_dtype,
        sub_core_grids=sub_core_grids,
    )

    torch_golden = torch.div(torch_input_a.float(), torch_input_b.float(), rounding_mode=rounding_mode)
    if output_dtype == ttnn.int32:
        torch_golden = torch_golden.to(torch.int32).float()

    assert output.dtype == output_dtype
    assert torch.equal(ttnn.to_torch(output).float(), torch_golden)


@pytest.mark.parametrize(
    "dtype_a, dtype_b",
    [
        (ttnn.int32, ttnn.bfloat16),
        (ttnn.bfloat16, ttnn.int32),
        (ttnn.int32, ttnn.float32),
        (ttnn.float32, ttnn.int32),
        (ttnn.bfloat16, ttnn.float32),
        (ttnn.float32, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize("rounding_mode", [None, "floor", "trunc"])
@pytest.mark.parametrize("output_dtype", [ttnn.float32, ttnn.bfloat16, ttnn.int32])
def test_div_mixed_input_dtypes_output_dtype(device, dtype_a, dtype_b, rounding_mode, output_dtype):
    # A mismatched 32-bit integer operand is promoted to the floating compute dtype before the
    # divide, so the requested output dtype has to survive that promotion rather than follow it.
    torch_input_a = torch.tensor([[7.0, -7.0, 6.0, -6.0]])
    torch_input_b = torch.full_like(torch_input_a, 2.0)

    def to_device(torch_tensor, dtype):
        torch_dtype = {ttnn.int32: torch.int32, ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[dtype]
        return ttnn.from_torch(torch_tensor.to(torch_dtype), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.div(
        to_device(torch_input_a, dtype_a),
        to_device(torch_input_b, dtype_b),
        rounding_mode=rounding_mode,
        dtype=output_dtype,
    )

    torch_golden = torch.div(torch_input_a, torch_input_b, rounding_mode=rounding_mode)
    if output_dtype == ttnn.int32:
        torch_golden = torch_golden.to(torch.int32)

    assert output.dtype == output_dtype
    assert torch.equal(ttnn.to_torch(output).float(), torch_golden.float())


def test_div_int32_rejects_non_float32_output_dtype(device, expect_error):
    torch_input_a = torch.tensor([[7, 6]], dtype=torch.int32)
    torch_input_b = torch.tensor([[2, 3]], dtype=torch.int32)

    input_tensor_a = ttnn.from_torch(torch_input_a, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_b, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    with expect_error(RuntimeError, "Incorrect output_dtype value for Integer Division"):
        ttnn.div(input_tensor_a, input_tensor_b, dtype=ttnn.int32)


@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize("scalar", [2.5, -3.14, 2.0, -2.0, 0.5, -0.5])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("use_sub_core_grids", [False, True])
def test_div_int32_float_scalar_promotion(device, rounding_mode, scalar, layout, use_sub_core_grids):
    # Format boundaries: FP32 has 24 significant bits, so +/- (2**24 + 3)
    # rounds on promotion; INT32 endpoints exercise the largest signed magnitudes.
    # The remaining small integers cover exact/inexact quotients and both signs.
    # Scalars +/-2.0 distinguish Python float promotion from integer dispatch;
    # +/-0.5 and +/-2.5 are binary-exact, while +/-3.14 also exercises FP32 rounding.
    torch_input = torch.tensor(
        [-(2**31), -(2**24 + 3), -1999, -7, -5, 0, 5, 7, 1999, 2**24 + 3, 2**31 - 1], dtype=torch.int32
    )
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=layout, device=device)
    cores = (
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 1))}) if use_sub_core_grids else None
    )
    result = ttnn.div(input_tensor, scalar, rounding_mode=rounding_mode, sub_core_grids=cores)
    expected = torch.div(torch_input.float(), scalar, rounding_mode=rounding_mode)
    assert result.dtype == ttnn.float32
    assert result.layout == layout
    # Match floating division's accuracy contract; rounded small results must be exact.
    actual = ttnn.to_torch(result)
    assert_with_ulp(expected_result=expected, actual_result=actual, ulp_threshold=1.0)
    if rounding_mode is not None:
        assert torch.equal(actual[2:9], expected[2:9])


@pytest.mark.parametrize("rounding_mode", ["trunc", "floor"])
def test_div_int32_integer_and_float_scalar_remain_distinct(device, rounding_mode):
    # 2**24 + 3 rounds to 2**24 + 4 in FP32, so dividing by 2 versus 2.0
    # exposes lost integer precision; the negative case also checks floor vs trunc.
    torch_input = torch.tensor([-(2**24 + 3), 2**24 + 3], dtype=torch.int32)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    integer_result = ttnn.div(input_tensor, 2, rounding_mode=rounding_mode)
    float_result = ttnn.div(input_tensor, 2.0, rounding_mode=rounding_mode)
    assert integer_result.dtype == ttnn.int32
    assert float_result.dtype == ttnn.float32
    assert torch.equal(ttnn.to_torch(integer_result), torch.div(torch_input, 2, rounding_mode=rounding_mode))
    assert torch.equal(ttnn.to_torch(float_result), torch.div(torch_input.float(), 2.0, rounding_mode=rounding_mode))


@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize("output_dtype", [ttnn.float32, ttnn.bfloat16, ttnn.int32])
@pytest.mark.parametrize("preallocated", [False, True])
def test_div_int32_float_scalar_promotion_output_dtype(device, rounding_mode, output_dtype, preallocated):
    torch_input = torch.tensor([-7, -5, 0, 5, 7], dtype=torch.int32)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = (
        ttnn.from_torch(torch.zeros_like(torch_input), dtype=output_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        if preallocated
        else None
    )
    result = ttnn.div(input_tensor, 2.0, rounding_mode=rounding_mode, dtype=output_dtype, output_tensor=output_tensor)
    torch_dtype = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.int32: torch.int32}[output_dtype]
    expected = torch.div(torch_input.float(), 2.0, rounding_mode=rounding_mode).to(torch_dtype)
    assert result.dtype == output_dtype
    assert torch.equal(ttnn.to_torch(result), expected)
    if preallocated:
        assert torch.equal(ttnn.to_torch(output_tensor), expected)


@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("use_sub_core_grids", [False, True])
@pytest.mark.parametrize("interleaved_output", [False, True])
@pytest.mark.parametrize(
    "strategy,shape,grid_end",
    [
        pytest.param(ttnn.ShardStrategy.HEIGHT, (128, 32), (3, 0), id="height"),
        pytest.param(ttnn.ShardStrategy.WIDTH, (32, 128), (3, 0), id="width"),
        pytest.param(ttnn.ShardStrategy.BLOCK, (64, 64), (1, 1), id="block"),
    ],
)
def test_div_int32_float_scalar_promotion_sharded(
    device, rounding_mode, layout, use_sub_core_grids, interleaved_output, strategy, shape, grid_end
):
    # HW layout coverage: 4096 elements make four 32x32 tiles/shards. The signed
    # endpoints and +/- (2**24 + 3) separately check INT32 range and FP32 rounding.
    torch_input = (torch.arange(4096, dtype=torch.int32) % 1024 - 512).reshape(1, 1, *shape)
    torch_input.flatten()[:4] = torch.tensor([-(2**31), 2**31 - 1, -(2**24 + 3), 2**24 + 3])
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(*grid_end))})
    memory_config = ttnn.create_sharded_memory_config(
        shape=(32, 32),
        core_grid=cores,
        strategy=strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    input_tensor = ttnn.from_torch(
        torch_input, dtype=ttnn.int32, layout=layout, memory_config=memory_config, device=device
    )
    output_memory_config = ttnn.DRAM_MEMORY_CONFIG if interleaved_output else memory_config
    result = ttnn.div(
        input_tensor,
        2.5,
        rounding_mode=rounding_mode,
        memory_config=output_memory_config,
        sub_core_grids=cores if use_sub_core_grids else None,
    )
    assert result.dtype == ttnn.float32
    assert result.layout == layout
    assert result.memory_config() == output_memory_config
    expected = torch.div(torch_input.float(), 2.5, rounding_mode=rounding_mode)
    assert_with_ulp(expected_result=expected, actual_result=ttnn.to_torch(result), ulp_threshold=1.0)


@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize("scalar", [0.0, -0.0, float("inf"), -float("inf"), float("nan")])
def test_div_int32_float_scalar_promotion_nonfinite(device, rounding_mode, scalar):
    torch_input = torch.tensor([-(2**31), -5, 0, 5], dtype=torch.int32)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.div(input_tensor, scalar, rounding_mode=rounding_mode)
    fp32_input = ttnn.from_torch(torch_input.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    expected = ttnn.to_torch(ttnn.div(fp32_input, scalar, rounding_mode=rounding_mode))
    assert result.dtype == ttnn.float32
    torch.testing.assert_close(ttnn.to_torch(result), expected, rtol=0, atol=0, equal_nan=True)
    golden = torch.div(torch_input.float(), scalar, rounding_mode=rounding_mode)
    # PyTorch's floor division by infinity differs from flooring the quotient;
    # the comparison against native FP32 above covers that existing behavior.
    if abs(scalar) != float("inf") or rounding_mode != "floor":
        torch.testing.assert_close(ttnn.to_torch(result), golden, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
def test_div_int32_float_scalar_promotion_fast_mode(device, rounding_mode):
    torch_input = torch.tensor([-(2**31), -5, 0, 5], dtype=torch.int32)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.div(input_tensor, 2.5, rounding_mode=rounding_mode, fast_and_approximate_mode=True)
    assert result.dtype == ttnn.float32
    expected = torch.div(torch_input.float(), 2.5, rounding_mode=rounding_mode)
    assert_with_ulp(expected_result=expected, actual_result=ttnn.to_torch(result), ulp_threshold=1.0)


def test_div_int32_float_scalar_promotion_output_guards(device, expect_error):
    torch_input = torch.tensor([-7, 7], dtype=torch.int32)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(1, 1))})
    with expect_error(
        RuntimeError, "Division output typecast on a restricted grid requires a tiled interleaved tensor"
    ):
        ttnn.div(input_tensor, 2.0, rounding_mode="floor", dtype=ttnn.int32, sub_core_grids=cores)
    output_tensor = ttnn.from_torch(
        torch_input.float(), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    with expect_error(RuntimeError, "Optional output tensor with Row Major input is not supported"):
        ttnn.div(input_tensor, 2.0, output_tensor=output_tensor)


@pytest.mark.parametrize("shape", [(1, 1, 32, 32)])
@pytest.mark.parametrize("value", [0.0, 2.0], ids=["zero_divisor", "nonzero_divisor"])
@pytest.mark.parametrize(
    "requested_memcfg, expected_memcfg",
    (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (None, ttnn.L1_MEMORY_CONFIG),
    ),
    ids=["explicit_DRAM", "unset_follows_input"],
)
def test_div_no_nan_scalar_honours_memory_config(device, shape, value, requested_memcfg, expected_memcfg):
    """The scalar div_no_nan overload must return in the requested memory config.

    Both returns dropped it: `zeros_like(input_a)` on the zero-divisor branch and
    `multiply(input_a, 1/value)` on the other, with `output_mem_config` commented out in
    the signature. Both branches are covered because they are separate return statements.

    The unset case is covered too, and separately per branch: the two reach their default
    through different machinery — `zeros_like` via `fill`'s fast path, `multiply` via the
    binary op's own `value_or(input_a.memory_config())` — so a divergence between them
    would not show up in the explicit-request case alone.

    The input is placed in L1 with DRAM requested; with matching configs the inherited
    and requested values coincide and the defect is invisible.
    """
    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=torch.bfloat16) + 1.0
    input_tensor = ttnn.from_torch(
        torch_input, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    output = (
        ttnn.div_no_nan(input_tensor, value)
        if requested_memcfg is None
        else ttnn.div_no_nan(input_tensor, value, memory_config=requested_memcfg)
    )

    assert (
        output.memory_config() == expected_memcfg
    ), f"divisor {value}, requested {requested_memcfg}: expected {expected_memcfg} but landed in {output.memory_config()}"


# The unary-scalar fmod kernel truncates |x| * (1/s) to get the quotient. When 1/s rounds up,
# that product can land on (or just past) an integer, so the truncated quotient comes out one
# too high and |x| - quotient * s goes negative; the sign is then overwritten by copysgn and a
# remainder that should be just under s is reported as ~0 instead. The guard that was meant to
# catch this compared the truncated value against the same rounded product it was derived from,
# which cannot be greater, so it never fired.
#
# The trigger is inputs one ULP below an exact multiple of the divisor. They are sparse, so a
# PCC or allclose comparison over a whole tile of random values does not see them; the existing
# scalar sweeps also draw from [-100, 100], where the quotient is too small to hit the rounding.
#
# Partially addresses #51441.
@pytest.mark.parametrize("scalar", [3.0, 7.0, 1.5, -3.0])
def test_fmod_scalar_just_below_exact_multiple_fp32(scalar, device):
    n = torch.arange(1, 4097, dtype=torch.float64)
    exact_multiples = (n * scalar).to(torch.float32)
    torch_input_tensor = torch.nextafter(exact_multiples, torch.zeros_like(exact_multiples))

    golden_function = ttnn.get_golden_function(ttnn.fmod)
    torch_output_tensor = golden_function(torch_input_tensor, scalar, device=device)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.to_torch(ttnn.fmod(input_tensor, scalar))

    # fmod is exact in binary floating point: every result is representable in the input format,
    # so the correct output is the golden value itself, not an approximation of it.
    assert torch.allclose(output_tensor, torch_output_tensor, atol=1e-3, rtol=0)
    # The defining contract, stated separately: the magnitude of the result is below the modulus.
    assert torch.all(output_tensor.abs() < abs(scalar))

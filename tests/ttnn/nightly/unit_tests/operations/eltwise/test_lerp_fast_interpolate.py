# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.lerp(..., fast_interpolate=True): the caller guarantees weight in [0, 1].

On Blackhole the flag selects the kernel (input - weight * input) + weight * end. No intermediate
overflows for weight in [0, 1], weight 0 returns input and weight 1 returns end bit-exact. Other
architectures ignore the flag and run the default kernel.
"""

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import ternary_inputs

FLT_MIN = 2.0**-126
WEIGHT_MODES = ["tensor", "scalar", "row_bcast", "col_bcast", "scalar_bcast"]
WEIGHT_SHAPES = {
    "row_bcast": lambda h, w: (1, w),
    "col_bcast": lambda h, w: (h, 1),
    "scalar_bcast": lambda h, w: (1, 1),
}

# Bound in ULP of max(|input|, |weight * (end - input)|), as for the default kernel.
TERM_ULP = 2.5
# Bottom range: a * weight and end * weight are dropped when the factors' exponents sum below -127,
# and input - weight * input flushes below 2^-126. Where the larger term is below 2^-100 the error is
# bounded absolutely instead, by 2 * FLT_MIN (measured worst case on the grid 1.9922 * FLT_MIN).
BOTTOM_RANGE = 2.0**-100
BOTTOM_ATOL = 2 * FLT_MIN


def _skip_unless_blackhole(device):
    if not ttnn.device.is_blackhole(device):
        pytest.skip("fast_interpolate has a kernel on Blackhole only")


def _ulp(magnitude, torch_dtype):
    """Spacing of torch_dtype at magnitude (float64), clamped to the finite normal range."""
    finfo = torch.finfo(torch_dtype)
    clamped = magnitude.abs().clamp(min=finfo.tiny, max=finfo.max)
    return torch.ldexp(torch.full_like(clamped, finfo.eps), torch.frexp(clamped).exponent - 1)


def _to_device(tensor, device):
    return ttnn.from_torch(tensor, layout=ttnn.TILE_LAYOUT, device=device)


def _weight_arg(device, weight, weight_mode, shape, torch_dtype):
    """The weight as passed to ttnn.lerp and as a float64 golden operand, rounded to the input dtype."""
    if weight_mode == "scalar":
        return weight, torch.tensor(weight, dtype=torch_dtype).double()
    weight_shape = WEIGHT_SHAPES.get(weight_mode, lambda h, w: (h, w))(*shape)
    torch_weight = torch.full(weight_shape, weight, dtype=torch_dtype)
    return _to_device(torch_weight, device), torch_weight.double()


def _lerp(device, torch_input, torch_end, weight, fast_interpolate=True):
    weight = _to_device(weight, device) if torch.is_tensor(weight) else weight
    return ttnn.to_torch(
        ttnn.lerp(
            _to_device(torch_input, device), _to_device(torch_end, device), weight, fast_interpolate=fast_interpolate
        )
    )


def _bits(x):
    return x.view(torch.int32 if x.dtype == torch.float32 else torch.int16)


@pytest.mark.parametrize("input_dtype", ["bfloat16", "float32"])
def test_lerp_fast_interpolate_grid(device, input_dtype):
    """The category1 ternary grid with weight restricted to [0, 1]: finite everywhere, within 2.5 ULP
    of the larger term, and within BOTTOM_ATOL where the larger term is below BOTTOM_RANGE."""
    _skip_unless_blackhole(device)
    torch_dtype = getattr(torch, input_dtype)
    a, b, w = (t.flatten() for t in torch.broadcast_tensors(*ternary_inputs(include_zero=True, dtype=torch_dtype)))
    unit_weight = (w >= 0) & (w <= 1)
    shape = (-1, 1024)
    a, b, w = (t[unit_weight].reshape(shape) for t in (a, b, w))

    result = _lerp(device, a, b, w)

    a64, b64, w64 = a.double(), b.double(), w.double()
    term = w64 * (b64 - a64)
    exact = a64 + term
    larger = torch.maximum(a64.abs(), term.abs())
    golden = exact.to(torch_dtype)
    assert torch.isfinite(result).all(), "weight in [0, 1] must not overflow"
    abs_error = (result.double() - exact).abs()
    ulp_error = abs_error / _ulp(larger, torch_dtype)
    # Both sides at zero: the golden flushed to zero too.
    ulp_error = torch.where((result == 0) & (golden.abs() < FLT_MIN), torch.zeros_like(ulp_error), ulp_error)
    bottom = larger < BOTTOM_RANGE
    over = ulp_error > TERM_ULP
    assert (over & bottom).any(), "bottom-range exception matched nothing, so it has gone stale"
    worst_bottom = abs_error[over & bottom].max().item()
    assert worst_bottom <= BOTTOM_ATOL, f"bottom range: |result - exact| reaches {worst_bottom / FLT_MIN:.4f} FLT_MIN"
    outside = over & ~bottom
    assert not outside.any(), (
        f"{int(outside.sum())} lanes exceed {TERM_ULP} ULP of the larger term, worst "
        f"{ulp_error[outside].max().item():.3f}"
    )


ENDPOINT_VALUES = {
    "float32": [3.4028234663852886e38, 2.0**127, 1.0, FLT_MIN, 0.0],
    "bfloat16": [3.3895313892515355e38, 2.0**127, 1.0, FLT_MIN, 0.0],
}


@pytest.mark.parametrize("weight", [0.0, 1.0])
@pytest.mark.parametrize("weight_mode", WEIGHT_MODES)
@pytest.mark.parametrize("input_dtype", ["bfloat16", "float32"])
def test_lerp_fast_interpolate_endpoints(device, weight, weight_mode, input_dtype):
    """weight 0 returns input and weight 1 returns end bit-exact, for every pair of ±FLT_MAX, ±2^127,
    ±1, ±FLT_MIN and 0."""
    _skip_unless_blackhole(device)
    torch_dtype = getattr(torch, input_dtype)
    magnitudes = torch.tensor(ENDPOINT_VALUES[input_dtype], dtype=torch.float64)
    values = torch.cat([magnitudes, -magnitudes[magnitudes != 0]])
    pairs = torch.cartesian_prod(values, values)
    shape = (32, 128)
    repeats = (shape[0] * shape[1] + len(pairs) - 1) // len(pairs)
    pairs = pairs.repeat(repeats, 1)[: shape[0] * shape[1]]
    torch_input = pairs[:, 0].reshape(shape).to(torch_dtype)
    torch_end = pairs[:, 1].reshape(shape).to(torch_dtype)
    ttnn_weight, _ = _weight_arg(device, weight, weight_mode, shape, torch_dtype)

    result = ttnn.to_torch(
        ttnn.lerp(_to_device(torch_input, device), _to_device(torch_end, device), ttnn_weight, fast_interpolate=True)
    )

    expected = torch_input if weight == 0.0 else torch_end
    assert torch.equal(_bits(result), _bits(expected))


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 0.75, 1.0, 0.3])
@pytest.mark.parametrize("weight_mode", WEIGHT_MODES)
@pytest.mark.parametrize("input_dtype", ["bfloat16", "float32"])
def test_lerp_fast_interpolate_large_opposite_sign(device, weight, weight_mode, input_dtype):
    """Opposite-sign inputs up to the dtype max, where end - input overflows: finite and within 1 ULP
    of max(|input|, |end|)."""
    _skip_unless_blackhole(device)
    torch_dtype = getattr(torch, input_dtype)
    shape = (64, 64)
    generator = torch.Generator().manual_seed(0)
    finfo_max = torch.finfo(torch_dtype).max
    exponent = torch.randint(100, 128, (2, *shape), generator=generator).double()
    magnitude = ((1.0 + torch.rand((2, *shape), generator=generator, dtype=torch.float64)) * 2.0**exponent).clamp(
        max=finfo_max
    )
    sign = torch.where(torch.arange(shape[0] * shape[1]) % 2 == 0, 1.0, -1.0).reshape(shape)
    torch_input = (sign * magnitude[0]).to(torch_dtype)
    torch_end = (-sign * magnitude[1]).to(torch_dtype)
    ttnn_weight, golden_weight = _weight_arg(device, weight, weight_mode, shape, torch_dtype)

    result = ttnn.to_torch(
        ttnn.lerp(_to_device(torch_input, device), _to_device(torch_end, device), ttnn_weight, fast_interpolate=True)
    )

    a64, b64 = torch_input.double(), torch_end.double()
    exact = (1.0 - golden_weight) * a64 + golden_weight * b64
    assert torch.isfinite(result).all(), "weight in [0, 1] must not overflow"
    error = (result.double() - exact).abs() / _ulp(torch.maximum(a64.abs(), b64.abs()), torch_dtype)
    assert error.max().item() <= 1.0, f"error reaches {error.max().item():.3f} ULP of max(|input|, |end|)"


def test_lerp_fast_interpolate_ring_attention(device):
    """Ring-attention merge: float32 accumulator input, bfloat16 end, a (B, H, S, 1) float32 weight in
    [0, 1] broadcast along the last dim, float32 output."""
    _skip_unless_blackhole(device)
    generator = torch.Generator().manual_seed(0)
    shape = (1, 8, 256, 128)
    accumulator = torch.randn(shape, generator=generator, dtype=torch.float32) * 4
    step = (torch.randn(shape, generator=generator, dtype=torch.float32) * 4).to(torch.bfloat16)
    # w = exp(l_i - L') with L' >= l_i, plus exact 0 and 1 rows.
    weight = torch.exp(-torch.rand((*shape[:-1], 1), generator=generator, dtype=torch.float32) * 20)
    weight[..., 0, :] = 0.0
    weight[..., 1, :] = 1.0

    result = ttnn.to_torch(
        ttnn.lerp(
            _to_device(accumulator, device),
            _to_device(step, device),
            _to_device(weight, device),
            fast_interpolate=True,
        )
    )

    assert result.dtype == torch.float32, f"expected float32 output, got {result.dtype}"
    a64, b64, w64 = accumulator.double(), step.double(), weight.double()
    exact = (1.0 - w64) * a64 + w64 * b64
    error = (result.double() - exact).abs() / _ulp(torch.maximum(a64.abs(), b64.abs()), torch.float32)
    assert error.max().item() <= 1.0, f"error reaches {error.max().item():.3f} ULP of max(|input|, |end|)"
    assert torch.equal(result[..., 0, :], accumulator[..., 0, :]), "weight 0 must return the accumulator"
    assert torch.equal(result[..., 1, :], step[..., 1, :].float()), "weight 1 must return the step"


@pytest.mark.parametrize("weight", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("input_dtype", ["bfloat16", "float32"])
def test_lerp_fast_interpolate_scalar_weight(device, weight, input_dtype):
    """The scalar-weight overload with fast_interpolate on random inputs: within 1 ULP of
    max(|input|, |end|), endpoints bit-exact."""
    _skip_unless_blackhole(device)
    torch_dtype = getattr(torch, input_dtype)
    generator = torch.Generator().manual_seed(1)
    shape = (64, 256)
    torch_input = (torch.randn(shape, generator=generator) * 100).to(torch_dtype)
    torch_end = (torch.randn(shape, generator=generator) * 100).to(torch_dtype)

    result = _lerp(device, torch_input, torch_end, weight)

    golden_weight = torch.tensor(weight, dtype=torch_dtype).double()
    a64, b64 = torch_input.double(), torch_end.double()
    exact = (1.0 - golden_weight) * a64 + golden_weight * b64
    error = (result.double() - exact).abs() / _ulp(torch.maximum(a64.abs(), b64.abs()), torch_dtype)
    assert error.max().item() <= 1.0, f"error reaches {error.max().item():.3f} ULP of max(|input|, |end|)"
    if weight in (0.0, 1.0):
        assert torch.equal(_bits(result), _bits(torch_input if weight == 0.0 else torch_end))


@pytest.mark.parametrize("use_scalar_weight", [True, False])
@pytest.mark.parametrize("input_dtype", ["bfloat16", "float32"])
def test_lerp_fast_interpolate_program_cache(device, use_scalar_weight, input_dtype):
    """The flag is part of the program hash: the same shapes with fast_interpolate off, on, off run
    the matching kernel each time. lerp(FLT_MIN, 1, 0) tells them apart: the default kernel flushes
    FLT_MIN / 2 and returns 0, the fast kernel returns FLT_MIN."""
    _skip_unless_blackhole(device)
    torch_dtype = getattr(torch, input_dtype)
    shape = (32, 64)
    torch_input = torch.full(shape, FLT_MIN, dtype=torch_dtype)
    torch_end = torch.ones(shape, dtype=torch_dtype)
    weight = 0.0 if use_scalar_weight else torch.zeros(shape, dtype=torch_dtype)
    default_expected = torch.zeros(shape, dtype=torch_dtype)

    entries = []
    for fast_interpolate, expected in ((False, default_expected), (True, torch_input), (False, default_expected)):
        result = _lerp(device, torch_input, torch_end, weight, fast_interpolate=fast_interpolate)
        assert torch.equal(result, expected), f"fast_interpolate={fast_interpolate} ran the wrong kernel"
        entries.append(device.num_program_cache_entries())
    assert entries[1] == entries[0] + 1, "fast_interpolate=True must add its own program"
    assert entries[2] == entries[1], "fast_interpolate=False must reuse the cached default program"


@pytest.mark.parametrize("input_dtype", ["bfloat16", "float32"])
def test_lerp_fast_interpolate_ignored_off_blackhole(device, capfd, input_dtype):
    """Off Blackhole the flag is ignored (with a warning): the result is the default kernel's."""
    if ttnn.device.is_blackhole(device):
        pytest.skip("Blackhole runs the fast_interpolate kernel")
    torch_dtype = getattr(torch, input_dtype)
    generator = torch.Generator().manual_seed(2)
    shape = (32, 64)
    torch_input = torch.randn(shape, generator=generator).to(torch_dtype)
    torch_end = torch.randn(shape, generator=generator).to(torch_dtype)
    weight = torch.rand(shape, generator=generator).to(torch_dtype)

    capfd.readouterr()
    fast = _lerp(device, torch_input, torch_end, weight, fast_interpolate=True)
    fast_log = "".join(capfd.readouterr())
    default = _lerp(device, torch_input, torch_end, weight, fast_interpolate=False)

    assert "fast_interpolate is implemented on Blackhole only" in fast_log, "expected the off-Blackhole warning"
    assert torch.equal(_bits(fast), _bits(default))


FALLBACK_WARNING = "fast_interpolate is not supported by the composite fallback"


def _fallback_inputs(device, fallback, use_scalar_weight):
    """Inputs that send ttnn.lerp to its composite fallback: INT32, or block float with a sub-tile
    (column) broadcast of end, or of the weight."""
    generator = torch.Generator().manual_seed(3)
    shape = (32, 64)
    if fallback == "int32":
        torch_input = torch.randint(-100, 100, shape, generator=generator, dtype=torch.int32)
        torch_end = torch.randint(-100, 100, shape, generator=generator, dtype=torch.int32)
        weight = torch.ones(shape, dtype=torch.int32)
        dtype = ttnn.int32
    else:
        torch_input = torch.randn(shape, generator=generator)
        # Scalar weight: end carries the column broadcast; tensor weight: the weight does.
        torch_end = torch.randn((shape[0], 1) if use_scalar_weight else shape, generator=generator)
        weight = 0.5 if use_scalar_weight else torch.rand((shape[0], 1), generator=generator)
        dtype = ttnn.bfloat8_b
    to_device = lambda t: ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return to_device(torch_input), to_device(torch_end), weight if use_scalar_weight else to_device(weight)


# INT32 with a scalar weight is left out: its composite fails on its own (INT32 + FLOAT32 add).
@pytest.mark.parametrize(
    "fallback, use_scalar_weight",
    [("int32", False), ("block_float_subtile_bcast", True), ("block_float_subtile_bcast", False)],
)
def test_lerp_fast_interpolate_fallback_warns(device, capfd, fallback, use_scalar_weight):
    """The composite fallback ignores fast_interpolate with a warning and returns the default result."""
    input_tensor, end_tensor, weight = _fallback_inputs(device, fallback, use_scalar_weight)

    capfd.readouterr()
    fast = ttnn.to_torch(ttnn.lerp(input_tensor, end_tensor, weight, fast_interpolate=True))
    fast_log = "".join(capfd.readouterr())
    default = ttnn.to_torch(ttnn.lerp(input_tensor, end_tensor, weight))
    default_log = "".join(capfd.readouterr())

    assert FALLBACK_WARNING in fast_log, "fast_interpolate=True on the fallback must log a warning"
    assert FALLBACK_WARNING not in default_log, "fast_interpolate=False must not warn"
    assert torch.equal(fast, default)

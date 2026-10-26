# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn

_INTEGER_EXPONENTS = [
    (32766.0, False),
    (32767.0, True),
    (32768.0, False),
    (65536.0, False),
    (-40000.0, False),
    (-32767.0, True),
    (4194305.0, True),
    (4194306.0, False),
    (8388609.0, True),
    (-4194305.0, True),
    (-8388609.0, True),
    (16777216.0, False),
]

_NON_INTEGER_EXPONENTS = [32768.5, 4194304.5]


_TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}


def _tile(value, dtype):
    return torch.full((1, 1, 32, 32), value, dtype=_TORCH_DTYPE[dtype])


def _to_device(device, value, dtype):
    return ttnn.from_torch(_tile(value, dtype), layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)


def _run(device, base, exponent, dtype=ttnn.float32):
    base_tt = _to_device(device, base, dtype)
    result = ttnn.pow(base_tt, _to_device(device, exponent, dtype))
    return ttnn.to_torch(result).double().reshape(-1)


def _allow_failure_on_quasar(request, device):
    if device.arch() == ttnn.device.Arch.QUASAR:
        request.applymarker(
            pytest.mark.xfail(reason="Quasar's pow kernel is fixed separately", raises=AssertionError, strict=False)
        )


@pytest.mark.parametrize("exponent, is_odd", _INTEGER_EXPONENTS)
def test_pow_negative_one_follows_the_exponent_parity(request, device, exponent, is_odd):
    if abs(exponent) >= 32768:
        _allow_failure_on_quasar(request, device)
    expected = -1.0 if is_odd else 1.0
    result = _run(device, -1.0, exponent)
    assert (result == expected).all(), (
        f"pow(-1, {exponent:g}): expected {expected:g} for an "
        f"{'odd' if is_odd else 'even'} exponent, got {result[0].item()}"
    )


_SATURATING_CASES = [
    (-2.0, 32768.0),
    (-2.0, 65536.0),
    (-2.0, 4194305.0),
    (-2.0, 4194306.0),
    (-2.0, 8388609.0),
    (-2.0, -40000.0),
    (-2.0, -4194305.0),
    (-0.5, 40000.0),
    (-0.5, -40000.0),
    (-0.5, -4194305.0),
]


@pytest.mark.parametrize("base, exponent", _SATURATING_CASES)
def test_pow_saturated_result_keeps_the_exponent_parity(request, device, base, exponent):
    _allow_failure_on_quasar(request, device)
    expected = torch.pow(torch.tensor(base, dtype=torch.float32), torch.tensor(exponent, dtype=torch.float32))
    result = _run(device, base, exponent)
    assert (result == expected.double()).all() and (torch.signbit(result) == torch.signbit(expected)).all(), (
        f"pow({base:g}, {exponent:g}): expected {expected.item()} "
        f"(signbit {int(torch.signbit(expected))}), got {result[0].item()} "
        f"(signbit {int(torch.signbit(result[0]))})"
    )


@pytest.mark.parametrize("exponent", _NON_INTEGER_EXPONENTS)
def test_pow_negative_base_non_integer_exponent_stays_nan(device, exponent):
    result = _run(device, -1.0, exponent)
    assert result.isnan().all(), f"pow(-1, {exponent:g}): expected nan, got {result[0].item()}"


@pytest.mark.parametrize("exponent", [32768.0, 4194306.0, 32768.5])
def test_pow_positive_base_is_unaffected(device, exponent):
    result = _run(device, 1.0, exponent)
    assert (result == 1.0).all(), f"pow(1, {exponent:g}): expected 1, got {result[0].item()}"


_BFLOAT16_EVEN_EXPONENTS = [32768.0, 33024.0, 65536.0, -65536.0, 16777216.0]


@pytest.mark.parametrize("exponent", _BFLOAT16_EVEN_EXPONENTS)
def test_pow_negative_one_follows_the_exponent_parity_bfloat16(request, device, exponent):
    _allow_failure_on_quasar(request, device)
    result = _run(device, -1.0, exponent, ttnn.bfloat16)
    assert (result == 1.0).all(), f"pow(-1, {exponent:g}) in bfloat16: expected 1, got {result[0].item()}"


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("exponent", [0.0, -0.0])
def test_pow_negative_base_zero_exponent_is_one(request, device, exponent, dtype):
    _allow_failure_on_quasar(request, device)
    result = _run(device, -2.0, exponent, dtype)
    assert (result == 1.0).all(), f"pow(-2, {exponent:g}): expected 1, got {result[0].item()}"


@pytest.mark.parametrize("exponent", [0.25, -0.25, 2.0**-31, -(2.0**-31), 2.0**-126, -(2.0**-126)])
def test_pow_negative_base_small_non_integer_exponent_is_nan(device, exponent):
    result = _run(device, -2.0, exponent)
    assert result.isnan().all(), f"pow(-2, {exponent:g}): expected nan, got {result[0].item()}"


@pytest.mark.parametrize("exponent", [32769.0, -32769.0, 16777215.0, -16777215.0])
def test_pow_negative_zero_odd_exponent_keeps_the_sign(request, device, exponent):
    _allow_failure_on_quasar(request, device)
    expected = torch.pow(torch.tensor(-0.0, dtype=torch.float32), torch.tensor(exponent, dtype=torch.float32)).item()
    result = _run(device, -0.0, exponent)
    assert (result == expected).all() and torch.signbit(
        result
    ).all(), f"pow(-0, {exponent:g}): expected {expected!r}, got {result[0].item()!r}"


@pytest.mark.parametrize("base", [-2.0, -1.0, float("-inf")])
@pytest.mark.parametrize("exponent", [float("nan"), -float("nan")])
def test_pow_negative_base_nan_exponent_is_nan(device, base, exponent):
    base_tt, exponent_tt = (
        ttnn.from_torch(
            _tile(value, ttnn.float32),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.float32,
            device=device,
            preserve_nan_values=True,
        )
        for value in (base, exponent)
    )
    result = ttnn.to_torch(ttnn.pow(base_tt, exponent_tt)).double().reshape(-1)
    assert torch.isnan(result).all(), f"pow({base:g}, {exponent}): expected nan, got {result[0].item()!r}"

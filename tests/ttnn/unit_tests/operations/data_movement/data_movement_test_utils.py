# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Inputs over each dtype's whole range and an exact comparison, for tests of ops that must not change values."""

import math

import torch
import ttnn

# torch dtype of the values, the signed int dtype with the same bits, and the number of mantissa bits.
_FLOATS = {
    ttnn.bfloat16: (torch.bfloat16, torch.int16, 7),
    ttnn.float32: (torch.float32, torch.int32, 23),
}
# torch dtype that ttnn.from_torch takes for the dtype, and the dtype's smallest and largest values.
_INTS = {
    ttnn.int32: (torch.int32, -(2**31), 2**31 - 1),
    ttnn.uint32: (torch.uint32, 0, 2**32 - 1),
    ttnn.uint16: (torch.uint16, 0, 2**16 - 1),
    ttnn.uint8: (torch.uint8, 0, 2**8 - 1),
}
_BLOCK_FLOATS = (ttnn.bfloat8_b, ttnn.bfloat4_b)


def full_range_input(shape, dtype, seed=0, nan=True):
    """Random values over ``dtype``'s whole range, as a torch tensor that ``ttnn.from_torch`` takes for ``dtype``.

    Floats get random signs, exponents and mantissas, so every normal exponent occurs. Their first elements are
    +inf, -inf, NaN (left out if ``nan`` is False), -0.0 and +0.0. Subnormals are left out, since the hardware may
    flush them to zero. Integers cover their whole range, and their first elements are their smallest and largest
    values. Block floats share one exponent per 16 values, so they get standard normal values instead.
    """
    torch.manual_seed(seed)
    if dtype in _FLOATS:
        torch_dtype, bits_dtype, mantissa_bits = _FLOATS[dtype]
        exponent_bits = torch.finfo(torch_dtype).bits - 1 - mantissa_bits
        exponent = torch.randint(1, 2**exponent_bits - 1, shape)  # all-zero and all-one exponents are not normal
        mantissa = torch.randint(0, 2**mantissa_bits, shape)
        magnitude = ((exponent << mantissa_bits) | mantissa).to(bits_dtype).view(torch_dtype)
        values = torch.where(torch.randint(0, 2, shape, dtype=torch.bool), -magnitude, magnitude)
        special = [math.inf, -math.inf] + ([math.nan] if nan else []) + [-0.0, 0.0]
        values.view(-1)[: len(special)] = torch.tensor(special, dtype=torch_dtype)
        return values
    if dtype in _INTS:
        torch_dtype, low, high = _INTS[dtype]
        values = torch.randint(low, high + 1, shape, dtype=torch.int64)
        values.view(-1)[:2] = torch.tensor([low, high])
        return values.to(torch_dtype)
    if dtype in _BLOCK_FLOATS:
        return torch.randn(shape, dtype=torch.float32)
    raise ValueError(f"no input for {dtype}")


def assert_exact(expected, actual, signed_zeros=True):
    """Same shape and the same values, compared as float64 so every dtype above compares exactly.

    NaN matches NaN, and -0.0 matches only -0.0, unless ``signed_zeros`` is False.
    """
    assert list(expected.shape) == list(actual.shape), f"shape {list(actual.shape)} != {list(expected.shape)}"
    expected, actual = expected.to(torch.float64), actual.to(torch.float64)
    same = expected == actual
    if signed_zeros:
        same &= expected.signbit() == actual.signbit()
    same |= expected.isnan() & actual.isnan()
    differ = (~same).nonzero().tolist()
    assert not differ, (
        f"{len(differ)} of {same.numel()} values differ; at {differ[0]}: "
        f"{actual[tuple(differ[0])].item()} != {expected[tuple(differ[0])].item()}"
    )

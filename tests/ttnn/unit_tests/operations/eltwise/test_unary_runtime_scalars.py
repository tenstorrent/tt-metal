# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Unary ops on the generic SFPU kernel take their scalar parameters as runtime args, so a new
value must reuse the cached program and still produce that value's result.
"""

import math

import pytest
import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.fixture
def isolate_program_cache(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


VALUES = [0.5, 1.25, 2.75, 3.5]

# name, ttnn call, torch golden, positive input
FLOAT_OPS = [
    ("pow", lambda x, v: ttnn.pow(x, v), lambda t, v: torch.pow(t, v), True),
    ("rpow", lambda x, v: ttnn.rpow(x, v), lambda t, v: torch.pow(v, t), True),
    ("rdiv", lambda x, v: ttnn.rdiv(x, v), lambda t, v: v / t, True),
    ("clamp", lambda x, v: ttnn.clamp(x, min=-v, max=v), lambda t, v: torch.clamp(t, -v, v), False),
    ("elu", lambda x, v: ttnn.elu(x, alpha=v), lambda t, v: F.elu(t, alpha=v), False),
    ("celu", lambda x, v: ttnn.celu(x, alpha=v), lambda t, v: F.celu(t, alpha=v), False),
    ("leaky_relu", lambda x, v: ttnn.leaky_relu(x, negative_slope=v / 10), lambda t, v: F.leaky_relu(t, v / 10), False),
    ("heaviside", lambda x, v: ttnn.heaviside(x, v), lambda t, v: torch.heaviside(t, torch.tensor(v)), False),
    ("maximum", lambda x, v: ttnn.maximum(x, v), lambda t, v: torch.maximum(t, torch.tensor(v)), False),
    ("minimum", lambda x, v: ttnn.minimum(x, v), lambda t, v: torch.minimum(t, torch.tensor(v)), False),
    ("fmod", lambda x, v: ttnn.fmod(x, v), lambda t, v: torch.fmod(t, v), False),
    ("remainder", lambda x, v: ttnn.remainder(x, v), lambda t, v: torch.remainder(t, v), False),
    ("threshold", lambda x, v: ttnn.threshold(x, v, -v), lambda t, v: F.threshold(t, v, -v), False),
    ("softplus", lambda x, v: ttnn.softplus(x, beta=v), lambda t, v: F.softplus(t, beta=v), False),
    ("hardshrink", lambda x, v: ttnn.hardshrink(x, lambd=v / 10), lambda t, v: F.hardshrink(t, v / 10), False),
    ("softshrink", lambda x, v: ttnn.softshrink(x, lambd=v / 10), lambda t, v: F.softshrink(t, v / 10), False),
]


@pytest.mark.parametrize("name, ttnn_op, torch_op, positive", FLOAT_OPS, ids=[op[0] for op in FLOAT_OPS])
def test_new_scalar_value_reuses_program(device, isolate_program_cache, name, ttnn_op, torch_op, positive):
    torch.manual_seed(0)
    torch_x = torch.rand(1, 1, 64, 96) + 0.5 if positive else torch.randn(1, 1, 64, 96) * 3
    torch_x = torch_x.to(torch.bfloat16)
    tt_x = ttnn.from_torch(torch_x, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_op(tt_x, VALUES[0])
    ttnn.synchronize_device(device)
    entries = device.num_program_cache_entries()

    for value in VALUES[1:] + VALUES[:1]:
        result = ttnn.to_torch(ttnn_op(tt_x, value))
        assert_with_pcc(torch_op(torch_x.float(), value), result.float(), 0.999)

    ttnn.synchronize_device(device)
    assert device.num_program_cache_entries() == entries


def test_fill_value_reuses_program(device, isolate_program_cache):
    tt_x = ttnn.from_torch(torch.randn(1, 1, 64, 96), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.fill(tt_x, VALUES[0])
    ttnn.synchronize_device(device)
    entries = device.num_program_cache_entries()

    for value in VALUES[1:] + [-7.25, 1e-3]:
        result = ttnn.to_torch(ttnn.fill(tt_x, value))
        assert torch.equal(result, torch.full_like(result, value))

    ttnn.synchronize_device(device)
    assert device.num_program_cache_entries() == entries


@pytest.mark.parametrize("first, second", [(0.0, -0.0), (-0.0, 0.0)])
def test_fill_keeps_sign_of_zero(device, isolate_program_cache, first, second):
    """+0.0 and -0.0 compare equal, so a cache keyed on the value used to hand the second call the first one's program."""
    tt_x = ttnn.from_torch(torch.randn(1, 1, 64, 96), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    for value in (first, second):
        result = ttnn.to_torch(ttnn.fill(tt_x, value))
        assert torch.equal(
            torch.signbit(result), torch.full_like(result, math.copysign(1.0, value) < 0, dtype=torch.bool)
        )


INT_OPS = [
    ("bitwise_and", lambda x, v: ttnn.bitwise_and(x, v), lambda t, v: torch.bitwise_and(t, v)),
    ("bitwise_or", lambda x, v: ttnn.bitwise_or(x, v), lambda t, v: torch.bitwise_or(t, v)),
    ("left_shift", lambda x, v: ttnn.bitwise_left_shift(x, v), lambda t, v: torch.bitwise_left_shift(t, v)),
]


@pytest.mark.parametrize("name, ttnn_op, torch_op", INT_OPS, ids=[op[0] for op in INT_OPS])
def test_new_int_scalar_value_reuses_program(device, isolate_program_cache, name, ttnn_op, torch_op):
    torch.manual_seed(0)
    torch_x = torch.randint(0, 1000, (1, 1, 64, 96), dtype=torch.int32)
    tt_x = ttnn.from_torch(torch_x, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_op(tt_x, 1)
    ttnn.synchronize_device(device)
    entries = device.num_program_cache_entries()

    for value in [3, 7, 13, 1]:
        result = ttnn.to_torch(ttnn_op(tt_x, value))
        assert torch.equal(result, torch_op(torch_x, value))

    ttnn.synchronize_device(device)
    assert device.num_program_cache_entries() == entries

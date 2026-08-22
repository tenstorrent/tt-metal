# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("shape", [(1, 1, 32, 32)])
def test_nextafter(device, shape):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape, dtype=torch.bfloat16)

    torch_output_tensor = torch.nextafter(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.nextafter(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


def _nextafter_fp32(device, vals_a, vals_b):
    """Run ttnn.nextafter on two FLOAT32 value lists, returned as a float32 torch tensor."""
    # Tile the row to 32 to fill a tile; only the first len(vals_a) columns are compared.
    a = torch.tensor(vals_a, dtype=torch.float32).reshape(1, -1).expand(32, -1).contiguous()
    b = torch.tensor(vals_b, dtype=torch.float32).reshape(1, -1).expand(32, -1).contiguous()
    ta = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.to_torch(ttnn.nextafter(ta, tb))[0, : len(vals_a)]


# On [1, 2) the FLOAT32 ULP is exactly hal::get_eps() (2^-23), so this is the one range where
# ttnn.nextafter agrees with torch.nextafter exactly rather than merely in direction. Values are
# kept strictly inside the binade in the downward cases: stepping below 1.0 halves the ULP, which
# the fixed eps step cannot follow.
FP32_EXACT = [
    pytest.param(1.0, 2.0, id="1.0_up"),
    pytest.param(1.25, 2.0, id="1.25_up"),
    pytest.param(1.5, 2.0, id="1.5_up"),
    pytest.param(1.9999998807907104, 2.0, id="max_binade_up"),
    pytest.param(1.25, 1.0, id="1.25_down"),
    pytest.param(1.5, 1.0, id="1.5_down"),
    pytest.param(1.75, 1.0, id="1.75_down"),
]


@pytest.mark.parametrize("a, b", FP32_EXACT)
def test_nextafter_fp32_matches_torch_in_unit_binade(device, a, b):
    """ttnn.nextafter must step TOWARDS input_b, matching torch.nextafter on [1, 2).

    Guards against the arms of the where-chain being swapped. The pre-existing bfloat16 PCC test
    above cannot see this: at every bfloat16 magnitude the eps step is far below one ULP, so the op
    returns input_a unchanged and PCC stays at ~1.0 whichever direction the arms select.
    """
    got = _nextafter_fp32(device, [a], [b])[0].item()
    expected = torch.nextafter(torch.tensor(a), torch.tensor(b)).item()
    assert got == expected, f"nextafter({a}, {b}) = {got!r}, expected {expected!r}"


@pytest.mark.parametrize("a", [1.0, 1.5, 0.5, 0.001, -1.0, -1.5, -0.001])
def test_nextafter_fp32_moves_towards_other(device, a):
    """The defining directional property, over magnitudes where the eps step is not absorbed.

    Magnitude is deliberately not asserted here -- away from [1, 2) the fixed eps step is not one
    ULP (it overshoots below 1.0), so only the direction is well-defined. |a| >= 2 is excluded
    because there eps is under half a ULP and rounds away entirely, leaving input_a unchanged.
    """
    got_up = _nextafter_fp32(device, [a], [a + 10.0])[0].item()
    got_down = _nextafter_fp32(device, [a], [a - 10.0])[0].item()

    assert got_up > a, f"nextafter({a}, {a + 10.0}) = {got_up!r} did not move up"
    assert got_down < a, f"nextafter({a}, {a - 10.0}) = {got_down!r} did not move down"


@pytest.mark.parametrize("a", [1.0, 1.5, -1.5, 0.0, 100.0])
def test_nextafter_fp32_equal_inputs_unchanged(device, a):
    """nextafter(a, a) == a -- neither where-chain arm may fire when the inputs are equal."""
    got = _nextafter_fp32(device, [a], [a])[0].item()
    assert got == a, f"nextafter({a}, {a}) = {got!r}, expected {a!r}"

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

# A BFLOAT8_B / BFLOAT4_B tile shares one exponent per 16-lane block of a row. At these widths the last block of each
# row holds real lanes and padding, and an op whose result at the zero padding is inf must not zero the real lanes.
# W=48 and W=64 are block aligned and are the controls.
WIDTHS = [1, 33, 40, 50, 63, 48, 64]

UNARY_OPS = {
    "reciprocal": (ttnn.reciprocal, torch.reciprocal),
    "rsqrt": (ttnn.rsqrt, torch.rsqrt),
    "log": (ttnn.log, torch.log),
    "log2": (ttnn.log2, torch.log2),
    "log10": (ttnn.log10, torch.log10),
}

# (atol, rtol) against the exact result of the stored inputs; the bfloat4_b bound is the error of the aligned controls.
TOLERANCE = {ttnn.bfloat8_b: (0.04, 0.04), ttnn.bfloat4_b: (0.2, 0.3)}


def _to_device(torch_tensor, dtype, device):
    return ttnn.from_torch(torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _assert_real_lanes_kept(result, expected, dtype):
    assert (result != 0).all(), f"{(result == 0).sum()} of {result.numel()} real lanes came back as 0"
    atol, rtol = TOLERANCE[dtype]
    torch.testing.assert_close(result, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b], ids=["bfloat8_b", "bfloat4_b"])
@pytest.mark.parametrize("width", WIDTHS, ids=lambda w: f"W={w}")
@pytest.mark.parametrize("op_name", list(UNARY_OPS))
def test_unary_block_float_partial_block_ignores_padding(device, op_name, width, dtype):
    ttnn_op, torch_op = UNARY_OPS[op_name]
    torch.manual_seed(0)
    torch_input = torch.empty(2, 32, width).uniform_(2.0, 4.0)
    input_tensor = _to_device(torch_input, dtype, device)

    result = ttnn.to_torch(ttnn_op(input_tensor)).float()
    expected = torch_op(ttnn.to_torch(input_tensor).float())

    _assert_real_lanes_kept(result, expected, dtype)


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b], ids=["bfloat8_b", "bfloat4_b"])
@pytest.mark.parametrize("width", WIDTHS, ids=lambda w: f"W={w}")
def test_div_block_float_partial_block_ignores_padding(device, width, dtype):
    torch.manual_seed(0)
    torch_b = torch.empty(2, 32, width).uniform_(2.0, 4.0)
    a = _to_device(torch.ones(2, 32, width), dtype, device)
    b = _to_device(torch_b, dtype, device)

    result = ttnn.to_torch(ttnn.div(a, b)).float()
    expected = 1.0 / ttnn.to_torch(b).float()

    _assert_real_lanes_kept(result, expected, dtype)


@pytest.mark.parametrize("width", [33, 50], ids=lambda w: f"W={w}")
def test_unary_block_float_partial_block_preallocated_output(device, width):
    torch.manual_seed(0)
    input_tensor = _to_device(torch.empty(32, width).uniform_(2.0, 4.0), ttnn.bfloat8_b, device)
    output_tensor = _to_device(torch.zeros(32, width), ttnn.bfloat8_b, device)

    ttnn.reciprocal(input_tensor, output_tensor=output_tensor)

    result = ttnn.to_torch(output_tensor).float()
    expected = torch.reciprocal(ttnn.to_torch(input_tensor).float())
    _assert_real_lanes_kept(result, expected, ttnn.bfloat8_b)

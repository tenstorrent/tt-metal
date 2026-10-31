# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device

OPS = {
    "add": (ttnn.add, torch.add),
    "subtract": (ttnn.subtract, torch.subtract),
    "multiply": (ttnn.multiply, torch.multiply),
    "divide": (ttnn.divide, torch.divide),
    "maximum": (ttnn.maximum, torch.maximum),
    "minimum": (ttnn.minimum, torch.minimum),
}


def _shapes(bcast, shape):
    n, c, h, w = shape
    full, row, col = shape, (n, c, 1, w), (n, c, h, 1)
    return {"row_b": (full, row), "row_a": (row, full), "row_a_col_b": (row, col), "row_b_col_a": (col, row)}[bcast]


def _bad_fraction(device, op, bcast, in_dtype, out_dtype, shape):
    ttnn_fn, torch_fn = OPS[op]
    torch.manual_seed(0)
    shape_a, shape_b = _shapes(bcast, shape)
    torch_a = torch.randn(shape_a)
    torch_b = torch.randn(shape_b)
    if op == "divide":
        torch_b = torch.where(torch_b >= 0, 0.5 + torch_b.abs(), -(0.5 + torch_b.abs()))
    a = ttnn.from_torch(torch_a, dtype=in_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch_b, dtype=in_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    expected = torch_fn(ttnn.to_torch(a).float(), ttnn.to_torch(b).float())
    kwargs = {} if out_dtype is None else {"dtype": out_dtype}
    out = ttnn.to_torch(ttnn_fn(a, b, **kwargs)).float()
    assert out.shape == expected.shape
    bad = ~torch.isfinite(out) | ((out - expected).abs() > 2e-2 + 2e-2 * expected.abs())
    return bad.float().mean().item()


@pytest.mark.parametrize("shape", [(1, 1, 64, 128), (1, 3, 50, 70)], ids=["64x128", "3x50x70"])
@pytest.mark.parametrize("in_dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b], ids=["bf16", "bfp8", "bfp4"])
@pytest.mark.parametrize("bcast", ["row_b", "row_a", "row_a_col_b", "row_b_col_a"])
@pytest.mark.parametrize("op", list(OPS))
def test_row_bcast_16bit_inputs_float32_output(device, op, bcast, in_dtype, shape):
    frac_bad = _bad_fraction(device, op, bcast, in_dtype, ttnn.float32, shape)
    assert frac_bad == 0.0, f"{op} {bcast} {in_dtype} -> float32 {shape}: {frac_bad:.1%} of the elements are wrong"


@pytest.mark.parametrize("bcast", ["row_b", "row_a"])
@pytest.mark.parametrize("op", ["add", "multiply", "maximum"])
def test_row_bcast_default_output_dtype(device, op, bcast):
    frac_bad = _bad_fraction(device, op, bcast, ttnn.bfloat16, None, (1, 1, 64, 128))
    assert frac_bad == 0.0, f"{op} {bcast} bfloat16 -> default dtype: {frac_bad:.1%} of the elements are wrong"

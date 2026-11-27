# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

NAN = float("nan")

FULL = (1, 1, 32, 32)
BCAST_SHAPES = {
    "full": FULL,
    "row": (1, 1, 1, 32),
    "col": (1, 1, 32, 1),
    "scalar": (1, 1, 1, 1),
}

# (id, ttnn op, torch golden, full-operand value, broadcast-operand value).
# Outputs are never NaN or -0.0, so the result only depends on whether the
# broadcast operand keeps its NaN / sign of zero on the way into the op.
B_CASES = [
    ("lt_nan", ttnn.lt, torch.lt, 1.0, NAN),
    ("div_negzero", ttnn.div, torch.div, 1.0, -0.0),
    ("atan2_negzero", ttnn.atan2, torch.atan2, 0.0, -0.0),
    (
        "isclose_nan",
        lambda a, b: ttnn.isclose(a, b, equal_nan=True),
        lambda a, b: torch.isclose(a, b, equal_nan=True),
        NAN,
        NAN,
    ),
]
A_CASES = [
    ("lt_nan", ttnn.lt, torch.lt, 1.0, NAN),
    ("atan2_negzero", ttnn.atan2, torch.atan2, -1.0, -0.0),
]


def _bits_equal(x, y):
    both_nan = torch.isnan(x) & torch.isnan(y)
    same = (x == y) & (torch.signbit(x) == torch.signbit(y))
    return both_nan | same


def _run(device, ttnn_op, torch_op, a, b, dtype):
    # torch.isclose does not broadcast a row against a column, so expand first.
    ga, gb = torch.broadcast_tensors(a.double(), b.double())
    golden = torch_op(ga, gb).double().to(a.dtype).double()
    ta = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(ttnn_op(ta, tb)).double()
    bad = (~_bits_equal(out, golden)).sum().item()
    assert (
        bad == 0
    ), f"{bad}/{out.numel()} mismatches, got {out.flatten()[0].item()!r} want {golden.flatten()[0].item()!r}"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("bshape", list(BCAST_SHAPES))
@pytest.mark.parametrize("case", B_CASES, ids=[c[0] for c in B_CASES])
def test_bcast_b_keeps_nan_and_negzero(device, case, bshape, dtype):
    _, ttnn_op, torch_op, full_v, bcast_v = case
    tdt = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    a = torch.full(FULL, full_v, dtype=tdt)
    b = torch.full(BCAST_SHAPES[bshape], bcast_v, dtype=tdt)
    _run(device, ttnn_op, torch_op, a, b, dtype)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("ashape", ["row", "col", "scalar"])
@pytest.mark.parametrize("case", A_CASES, ids=[c[0] for c in A_CASES])
def test_bcast_a_keeps_nan_and_negzero(device, case, ashape, dtype):
    _, ttnn_op, torch_op, full_v, bcast_v = case
    tdt = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    a = torch.full(BCAST_SHAPES[ashape], bcast_v, dtype=tdt)
    b = torch.full(FULL, full_v, dtype=tdt)
    _run(device, ttnn_op, torch_op, a, b, dtype)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_row_a_col_b_keeps_nan(device, dtype):
    tdt = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    a = torch.full(BCAST_SHAPES["row"], NAN, dtype=tdt)
    b = torch.full(BCAST_SHAPES["col"], NAN, dtype=tdt)
    _run(
        device,
        lambda x, y: ttnn.isclose(x, y, equal_nan=True),
        lambda x, y: torch.isclose(x, y, equal_nan=True),
        a,
        b,
        dtype,
    )

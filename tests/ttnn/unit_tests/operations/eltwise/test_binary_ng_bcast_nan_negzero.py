# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""A broadcast operand must reach the op with its NaN and its sign of zero intact.

Two ways the LLK subtile broadcast (row, column, scalar) used to break that for bf16:
  * NaN arrived as +inf (an intermediate bf16 pack of the expanded tile).
  * -0.0 arrived as +0.0 (that same pack, and ELWADD / the Src zero flag inside the broadcast).

The final bf16 pack also turns a NaN output into inf and a -0.0 output into +0.0, so the
discriminating cases pick ops whose output stays finite or becomes +-inf: lt(x, NaN) is False
but lt(x, inf) is True, and 1 / -0.0 is -inf but 1 / +0.0 is +inf.
"""

import math

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
    # gt, not lt: lt(NaN, 1) and lt(inf, 1) are both False.
    ("gt_nan", ttnn.gt, torch.gt, 1.0, NAN),
    ("atan2_negzero", ttnn.atan2, torch.atan2, -1.0, -0.0),
]


def _torch_dtype(dtype):
    return torch.float32 if dtype == ttnn.float32 else torch.bfloat16


def _pattern(shape, values, dtype=torch.bfloat16):
    """Cycles `values` over `shape` in row-major order, so each value lands at several distinct positions."""
    vals = torch.tensor(values, dtype=torch.float32)
    idx = torch.arange(math.prod(shape)) % len(vals)
    return vals[idx].reshape(shape).to(dtype)


def _assert_match(out, golden, *, packed_specials=False, inputs=()):
    """Bit-exact comparison (NaN == NaN, -0.0 != +0.0) that names the first mismatch.

    packed_specials: the output itself would be NaN or -0.0, which the final bf16 pack cannot keep.
    Then a NaN golden only requires a non-finite output and a zero golden only requires a zero.
    """
    out = out.double()
    golden = golden.double()
    if packed_specials:
        ok = torch.where(
            torch.isnan(golden),
            ~torch.isfinite(out),
            torch.where(golden == 0, out == 0, out == golden),
        )
    else:
        both_nan = torch.isnan(out) & torch.isnan(golden)
        ok = both_nan | ((out == golden) & (torch.signbit(out) == torch.signbit(golden)))
    bad = (~ok).sum().item()
    if bad:
        first = tuple(int(i) for i in (~ok).nonzero()[0])
        ins = ", ".join(f"in{k}={t.expand(out.shape)[first].item()!r}" for k, t in enumerate(inputs))
        raise AssertionError(
            f"{bad}/{out.numel()} mismatches; first at {first}: "
            f"got {out[first].item()!r} want {golden[first].item()!r} ({ins})"
        )


def _golden(torch_op, *operands, dtype=torch.bfloat16):
    # torch.isclose does not broadcast a row against a column, so expand first.
    expanded = torch.broadcast_tensors(*(t.double() for t in operands))
    return torch_op(*expanded).double().to(dtype)


def _to_device(t, device, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _run(device, ttnn_op, torch_op, a, b, dtype, *, out_dtype=None, packed_specials=False, **kwargs):
    golden = _golden(torch_op, a, b, dtype=_torch_dtype(out_dtype or dtype))
    out = ttnn.to_torch(ttnn_op(_to_device(a, device, dtype), _to_device(b, device, dtype), **kwargs))
    _assert_match(out, golden, packed_specials=packed_specials, inputs=(a, b))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("bshape", list(BCAST_SHAPES))
@pytest.mark.parametrize("case", B_CASES, ids=[c[0] for c in B_CASES])
def test_bcast_b_keeps_nan_and_negzero(device, case, bshape, dtype):
    _, ttnn_op, torch_op, full_v, bcast_v = case
    tdt = _torch_dtype(dtype)
    a = torch.full(FULL, full_v, dtype=tdt)
    b = torch.full(BCAST_SHAPES[bshape], bcast_v, dtype=tdt)
    _run(device, ttnn_op, torch_op, a, b, dtype)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("ashape", ["row", "col", "scalar"])
@pytest.mark.parametrize("case", A_CASES, ids=[c[0] for c in A_CASES])
def test_bcast_a_keeps_nan_and_negzero(device, case, ashape, dtype):
    _, ttnn_op, torch_op, full_v, bcast_v = case
    tdt = _torch_dtype(dtype)
    a = torch.full(BCAST_SHAPES[ashape], bcast_v, dtype=tdt)
    b = torch.full(FULL, full_v, dtype=tdt)
    _run(device, ttnn_op, torch_op, a, b, dtype)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_row_a_col_b_keeps_nan(device, dtype):
    tdt = _torch_dtype(dtype)
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


# ---- Multi-tile: several broadcast repeats per core (freq > 1) and cores that start mid-repeat ----

MT_FULL = (1, 2, 96, 64)
MT_BCAST = {
    "row": (1, 2, 1, 64),
    "col": (1, 2, 96, 1),
    "scalar": (1, 1, 1, 1),
}
# Finite numerators and power-of-two denominators keep every quotient exact; a zero denominator gives +-inf.
# A scalar-shaped b holds only the first value, so -0.0 goes first.
NUMERATORS = [1.0, -3.0, 6.0, 0.25, -1.5]
DENOMINATORS = [-0.0, 2.0, -4.0, 0.0, 0.5, -0.0, 1.0]
NAN_MIX = [NAN, 1.0, -2.0, NAN, 0.5, -0.0, 3.0]
COMPARANDS = [0.5, -3.0, 2.0, 1.0, -0.25]


def _one_core():
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})


@pytest.mark.parametrize("cores", ["all", "one"])
@pytest.mark.parametrize("bkind", list(MT_BCAST))
@pytest.mark.parametrize(
    "op_id, ttnn_op, torch_op, full_vals, bcast_vals",
    [
        ("lt_nan", ttnn.lt, torch.lt, COMPARANDS, NAN_MIX),
        ("div_negzero", ttnn.div, torch.div, NUMERATORS, DENOMINATORS),
    ],
    ids=["lt_nan", "div_negzero"],
)
def test_multi_tile_bcast_b(device, op_id, ttnn_op, torch_op, full_vals, bcast_vals, bkind, cores):
    a = _pattern(MT_FULL, full_vals)
    b = _pattern(MT_BCAST[bkind], bcast_vals)
    kwargs = {"sub_core_grids": _one_core()} if cores == "one" else {}
    _run(device, ttnn_op, torch_op, a, b, ttnn.bfloat16, **kwargs)


@pytest.mark.parametrize("akind", list(MT_BCAST))
@pytest.mark.parametrize(
    "op_id, ttnn_op, torch_op, bcast_vals, full_vals, cores",
    [
        ("gt_nan", ttnn.gt, torch.gt, NAN_MIX, COMPARANDS, "all"),
        ("gt_nan", ttnn.gt, torch.gt, NAN_MIX, COMPARANDS, "one"),
        # atan2(+-0, x < 0) is +-pi. ttnn.atan2 takes no sub_core_grids, so it runs on all cores only.
        ("atan2_negzero", ttnn.atan2, torch.atan2, [-0.0, 0.0, -0.0, -0.0, 0.0], [-1.0, -2.0, -0.5], "all"),
    ],
    ids=["gt_nan-cores=all", "gt_nan-cores=one", "atan2_negzero-cores=all"],
)
def test_multi_tile_bcast_a(device, op_id, ttnn_op, torch_op, bcast_vals, full_vals, cores, akind):
    a = _pattern(MT_BCAST[akind], bcast_vals)
    b = _pattern(MT_FULL, full_vals)
    kwargs = {"sub_core_grids": _one_core()} if cores == "one" else {}
    _run(device, ttnn_op, torch_op, a, b, ttnn.bfloat16, **kwargs)


@pytest.mark.parametrize("orientation", ["row_a_col_b", "row_b_col_a"])
def test_multi_tile_row_col(device, orientation):
    row = (1, 1, 1, 64)
    col = (1, 1, 96, 1)
    if orientation == "row_a_col_b":
        a, b = _pattern(row, COMPARANDS), _pattern(col, NAN_MIX)
    else:
        a, b = _pattern(col, COMPARANDS), _pattern(row, NAN_MIX)
    _run(device, ttnn.lt, torch.lt, a, b, ttnn.bfloat16)


# ---- ttnn.add: the output itself would be NaN / -0.0, which the final bf16 pack cannot keep ----


@pytest.mark.parametrize(
    "fast_and_approximate_mode",
    # True is ttnn.add's default (and what an unset C++ flag resolves to); is_binary_sfpu_op maps it to the
    # FPU kernels for bf16. False selects the SFPU kernels.
    [True, False],
    ids=["fpu_default", "sfpu"],
)
@pytest.mark.parametrize("bkind", list(MT_BCAST))
def test_add_bcast_records_packed_specials(device, bkind, fast_and_approximate_mode):
    """Records behaviour only: it cannot tell a lost NaN / -0.0 apart from a kept one.

    NaN + x packs to a non-finite value and -0.0 + -0.0 packs to a zero either way, so only the
    finite lanes are checked bit-exactly.
    """
    a = _pattern(MT_FULL, [1.0, -0.0, 2.5, -4.0, -0.0])
    b = _pattern(MT_BCAST[bkind], [NAN, -0.0, 0.5, -0.0, 3.0, NAN])
    _run(
        device,
        ttnn.add,
        torch.add,
        a,
        b,
        ttnn.bfloat16,
        packed_specials=True,
        fast_and_approximate_mode=fast_and_approximate_mode,
    )


# ---- Pre-activations: an activation on the broadcast operand moves the op off the LLK broadcast ----


@pytest.mark.parametrize("side", ["bcast", "other", "both"])
@pytest.mark.parametrize("bkind", list(MT_BCAST))
def test_bcast_with_pre_activation(device, bkind, side):
    # div(a, b) with b broadcast; ABS on the broadcast side turns its -0.0 into +0.0 and flips a's quotient
    # from -inf to +inf, so a dropped or misplaced activation shows up.
    a = _pattern(MT_FULL, [-1.0, 3.0, -6.0, 0.25, 1.5])
    b = _pattern(MT_BCAST[bkind], [2.0, -0.0, -4.0, 0.0, -0.5, -0.0])
    abs_b = side in ("bcast", "both")
    abs_a = side in ("other", "both")
    golden = _golden(torch.div, a.abs() if abs_a else a, b.abs() if abs_b else b)
    activation = [ttnn.UnaryWithParam(ttnn.UnaryOpType.ABS)]
    out = ttnn.div(
        _to_device(a, device, ttnn.bfloat16),
        _to_device(b, device, ttnn.bfloat16),
        input_tensor_a_activations=activation if abs_a else [],
        input_tensor_b_activations=activation if abs_b else [],
    )
    _assert_match(ttnn.to_torch(out), golden, inputs=(a, b))


@pytest.mark.parametrize("side", ["bcast", "other"])
def test_lt_nan_with_pre_activation(device, side):
    # ABS on the full operand keeps the LLK broadcast of b, so the NaN must still arrive as NaN. ABS on the
    # broadcast operand takes the reader-broadcast path, whose activation pass packs NaN to inf, so that
    # case uses finite values only.
    a = _pattern(FULL, [-1.0, 3.0, -0.5, 2.0])
    b = _pattern(BCAST_SHAPES["row"], [NAN, 1.0, NAN, -2.0])
    if side == "bcast":
        b = b.nan_to_num(nan=-7.0)
    activation = [ttnn.UnaryWithParam(ttnn.UnaryOpType.ABS)]
    golden = _golden(torch.lt, a.abs() if side == "other" else a, b.abs() if side == "bcast" else b)
    out = ttnn.lt(
        _to_device(a, device, ttnn.bfloat16),
        _to_device(b, device, ttnn.bfloat16),
        input_tensor_a_activations=activation if side == "other" else [],
        input_tensor_b_activations=activation if side == "bcast" else [],
    )
    _assert_match(ttnn.to_torch(out), golden, inputs=(a, b))


# ---- where: the output is one of the inputs, so NaN / -0.0 outputs cannot survive the bf16 pack ----
# These check the DEST slot layout of the where broadcast kernels bit-exactly on finite lanes; a NaN or
# -0.0 selected into the output is only checked as non-finite / zero (see _assert_match).

COND_VALS = [1.0, 0.0, NAN, -0.0, -2.0, 0.0]
TRUE_VALS = [1.5, -2.0, 3.0, -0.0, NAN, 7.0, -0.25]
FALSE_VALS = [-8.0, 0.5, NAN, -0.0, 4.0]


def _where_golden(c, t, f):
    return _golden(lambda c_, t_, f_: torch.where(c_ != 0, t_, f_), c, t, f)


def _where_check(device, c, t, f):
    """t or f may be a Python float (TTS / TST)."""
    tc = _to_device(c, device, ttnn.bfloat16)
    tt_ = _to_device(t, device, ttnn.bfloat16) if isinstance(t, torch.Tensor) else t
    tf = _to_device(f, device, ttnn.bfloat16) if isinstance(f, torch.Tensor) else f
    gt_ = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.bfloat16)
    gf = f if isinstance(f, torch.Tensor) else torch.tensor(f, dtype=torch.bfloat16)
    golden = _where_golden(c, gt_, gf)
    out = ttnn.to_torch(ttnn.where(tc, tt_, tf))
    _assert_match(out, golden, packed_specials=True, inputs=(c, gt_, gf))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_where_negzero_cond_is_false(device, dtype):
    # No broadcast: the where SFPU kernel itself must treat a -0.0 condition as false.
    torch_dtype = _torch_dtype(dtype)
    c = _pattern(FULL, [-0.0, 0.0, 1.0, -0.0, -2.0], dtype=torch_dtype)
    t = torch.full(FULL, 3.0, dtype=torch_dtype)
    f = torch.full(FULL, 4.0, dtype=torch_dtype)
    out = ttnn.where(_to_device(c, device, dtype), _to_device(t, device, dtype), _to_device(f, device, dtype))
    _assert_match(ttnn.to_torch(out), torch.where(c != 0, t, f), inputs=(c, t, f))


@pytest.mark.parametrize("bcast", ["cond", "true", "false"])
def test_where_ttt_row_bcast(device, bcast):
    c = _pattern(BCAST_SHAPES["row"] if bcast == "cond" else FULL, COND_VALS)
    t = _pattern(BCAST_SHAPES["row"] if bcast == "true" else FULL, TRUE_VALS)
    f = _pattern(BCAST_SHAPES["row"] if bcast == "false" else FULL, FALSE_VALS)
    _where_check(device, c, t, f)


@pytest.mark.parametrize("bcast", ["cond", "tensor"])
@pytest.mark.parametrize("variant", ["tts", "tst"])
def test_where_scalar_row_bcast(device, variant, bcast):
    c = _pattern(BCAST_SHAPES["row"] if bcast == "cond" else FULL, COND_VALS)
    tensor = _pattern(BCAST_SHAPES["row"] if bcast == "tensor" else FULL, TRUE_VALS)
    if variant == "tts":
        _where_check(device, c, tensor, 9.0)
    else:
        _where_check(device, c, 9.0, tensor)


@pytest.mark.parametrize("orientation", ["row_cond_col_tensor", "col_cond_row_tensor"])
@pytest.mark.parametrize("variant", ["tts", "tst"])
def test_where_scalar_row_col_bcast(device, variant, orientation):
    row, col = BCAST_SHAPES["row"], BCAST_SHAPES["col"]
    cond_shape, tensor_shape = (row, col) if orientation == "row_cond_col_tensor" else (col, row)
    c = _pattern(cond_shape, COND_VALS)
    tensor = _pattern(tensor_shape, TRUE_VALS)
    if variant == "tts":
        _where_check(device, c, tensor, 9.0)
    else:
        _where_check(device, c, 9.0, tensor)


# ---- Ternary SFPU TTT row broadcast (ternary_sfpu_row_bcast_ttt) with a discriminating input ----


def test_snake_beta_row_bcast_keeps_negzero(device):
    # x + sin^2(alpha * x) / beta with beta = -0.0 is -inf, and +inf for beta = +0.0. Only +-0 betas are used
    # so every output is exact. If this fails with +inf everywhere, check the SFPU reciprocal's handling of
    # -0.0 before the broadcast.
    x = torch.full(FULL, 1.0, dtype=torch.bfloat16)
    alpha = torch.full(BCAST_SHAPES["row"], 1.0, dtype=torch.bfloat16)
    beta = _pattern(BCAST_SHAPES["row"], [-0.0, 0.0, -0.0])
    golden = _golden(lambda x_, a_, b_: x_ + torch.sin(a_ * x_) ** 2 / b_, x, alpha, beta)
    out = ttnn.snake_beta(
        _to_device(x, device, ttnn.bfloat16),
        _to_device(alpha, device, ttnn.bfloat16),
        _to_device(beta, device, ttnn.bfloat16),
    )
    _assert_match(ttnn.to_torch(out), golden, inputs=(x, alpha, beta))

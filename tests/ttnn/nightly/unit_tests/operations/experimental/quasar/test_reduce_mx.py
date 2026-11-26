# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.sum / mean / max over MX (OCP microscaling) inputs on Quasar (issue #62742).

The tiled H / W reduce kernels unpack MX to Float16_b, reduce on the FPU and pack the result back to the input's
MX format (the reductions have no output dtype argument). Calls the native kernels do not cover (implicit tile
padding, the N / C axes, keepdim=False, ttnn.min, a full reduction) reduce in bfloat16 and are packed back to MX; see
the "outside the native kernels" tests.

MXFP4 column (H) sums: the LLK moves an MxFp4 data operand to the 2x-packed src format only when the scaler is
MXFP4 too (#59597); these factories keep a Float16_b scaler, so MXFP4 column sums take the plain path like every
other MX format.

Known gaps (see issue #62742): reductions over both H and W in one call (wrong on Quasar for bf16 too, skipped
below); std / var (the welford op rejects Quasar for every dtype). ttnn.min cannot use the fused -MAX(-x) kernels
with MX (their copy out of a scratch ring returns zeros), so it takes the bfloat16 route too.
"""

import pytest
import torch

import ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.quasar.mx_test_utils import (
    check,
    mx_round_trip,
    pcc_floor,
    skip_unless_quasar,
    to_device,
)

SHAPE = (1, 1, 64, 96)
BIG_SHAPE = (1, 2, 128, 256)  # several tiles per core on every core of the 8x4 simulator grid
TALL_SHAPE = (1, 1, 640, 32)  # Ht=20 with one column group (the bf16 H-split trigger); MX runs the plain H reduce
TWO_DTYPES = [ttnn.mxint8, ttnn.mxfp8_e4m3]

_OPS = {
    "sum": (ttnn.sum, lambda x, d: x.sum(d, keepdim=True)),
    "mean": (ttnn.mean, lambda x, d: x.mean(d, keepdim=True)),
    "max": (ttnn.max, lambda x, d: x.amax(d, keepdim=True)),
}


@pytest.fixture(autouse=True)
def _quasar_only(device):
    skip_unless_quasar(device)


@pytest.mark.parametrize("dtype", ttnn.MX_DTYPES)
@pytest.mark.parametrize("dim", [-1, -2], ids=["W", "H"])
@pytest.mark.parametrize("op", list(_OPS))
def test_reduce_mx(device, dtype, dim, op):
    torch.manual_seed(0)
    x = torch.randn(SHAPE)
    tt_op, torch_op = _OPS[op]
    out = tt_op(to_device(device, x, dtype), dim=dim, keepdim=True)
    assert out.dtype == dtype
    golden = mx_round_trip(torch_op(mx_round_trip(x, dtype), dim), dtype)
    check(golden, ttnn.to_torch(out), pcc_floor(dtype))


def _fallback_golden(x, dtype, torch_op, **kwargs):
    """The reduction of the MX values of `x`, packed back to the input's MX format (the bfloat16 route's result)."""
    return mx_round_trip(torch_op(mx_round_trip(x, dtype), **kwargs), dtype)


def _check_fallback(out, golden, dtype):
    assert out.dtype == dtype, f"output dtype {out.dtype}, expected {dtype}"
    assert out.layout == ttnn.TILE_LAYOUT, f"output layout {out.layout}"
    assert tuple(out.shape) == tuple(golden.shape), f"output shape {tuple(out.shape)}, expected {tuple(golden.shape)}"
    check(golden, ttnn.to_torch(out), pcc_floor(dtype))


# Outside the native kernels: the op reduces these in bfloat16 and packs the result back to MX.
@pytest.mark.parametrize("dtype", TWO_DTYPES)
@pytest.mark.parametrize("keepdim", [True, False], ids=["keepdim", "squeeze"])
@pytest.mark.parametrize("dim", [0, 1], ids=["N", "C"])
@pytest.mark.parametrize("op", ["sum", "max"])
def test_reduce_mx_nc_axis(device, dtype, dim, keepdim, op):
    torch.manual_seed(4)
    x = torch.randn(2, 2, 32, 32)
    tt_op, _ = _OPS[op]
    torch_fn = torch.sum if op == "sum" else (lambda t, dim, keepdim: t.amax(dim, keepdim))
    out = tt_op(to_device(device, x, dtype), dim=dim, keepdim=keepdim)
    _check_fallback(out, _fallback_golden(x, dtype, torch_fn, dim=dim, keepdim=keepdim), dtype)


@pytest.mark.parametrize("dtype", TWO_DTYPES)
@pytest.mark.parametrize("dim", [-1, -2], ids=["W", "H"])
@pytest.mark.parametrize("op", ["sum", "max"])
def test_reduce_mx_implicit_padding(device, dtype, dim, op):
    # H = 40 is not a tile multiple: the fill kernel for the padding does not know the MX layout.
    torch.manual_seed(5)
    x = torch.randn(1, 1, 40, 32)
    tt_op, torch_op = _OPS[op]
    out = tt_op(to_device(device, x, dtype), dim=dim, keepdim=True)
    _check_fallback(out, _fallback_golden(x, dtype, lambda t, dim: torch_op(t, dim), dim=dim), dtype)


@pytest.mark.parametrize("dtype", TWO_DTYPES)
@pytest.mark.parametrize("dim", [-1, -2], ids=["W", "H"])
@pytest.mark.parametrize("op", ["sum", "max"])
def test_reduce_mx_multi_tile(device, dtype, dim, op):
    torch.manual_seed(1)
    x = torch.randn(BIG_SHAPE)
    tt_op, torch_op = _OPS[op]
    out = tt_op(to_device(device, x, dtype), dim=dim, keepdim=True)
    golden = mx_round_trip(torch_op(mx_round_trip(x, dtype), dim), dtype)
    check(golden, ttnn.to_torch(out), pcc_floor(dtype))


@pytest.mark.parametrize("dtype", TWO_DTYPES)
@pytest.mark.parametrize("dim", [-1, -2], ids=["W", "H"])
def test_sum_mx_scalar(device, dtype, dim):
    # scalar != 1 takes the post-multiply path (mul_unary_tile on the reduced tile before the pack).
    torch.manual_seed(2)
    x = torch.randn(SHAPE)
    out = ttnn.sum(to_device(device, x, dtype), dim=dim, keepdim=True, scalar=2.0)
    golden = mx_round_trip(2.0 * mx_round_trip(x, dtype).sum(dim, keepdim=True), dtype)
    check(golden, ttnn.to_torch(out), pcc_floor(dtype))


@pytest.mark.parametrize("dtype", TWO_DTYPES)
@pytest.mark.parametrize("op", ["sum", "mean"])
def test_reduce_mx_tall_h(device, dtype, op):
    # A tall H with one column group would take the TILE H-axis split for bf16 / block-float, but that split is
    # rejected on Quasar (its stage-1 reader self-loops a DFB, dfb.cpp:603), so MX stays on the plain H reduce.
    torch.manual_seed(3)
    x = torch.randn(TALL_SHAPE)
    tt_op, torch_op = _OPS[op]
    out = tt_op(to_device(device, x, dtype), dim=-2, keepdim=True)
    golden = mx_round_trip(torch_op(mx_round_trip(x, dtype), -2), dtype)
    check(golden, ttnn.to_torch(out), pcc_floor(dtype))


@pytest.mark.parametrize("dtype", TWO_DTYPES)
@pytest.mark.parametrize("dim", [-1, -2], ids=["W", "H"])
def test_min_mx(device, dtype, dim):
    # ttnn.min lowers to -MAX(-x) through the fused-negate kernels, whose copy out of the scratch ring packs zeros for
    # an MX input on Quasar (issue #62742), so MX takes the bfloat16 route.
    torch.manual_seed(6)
    x = torch.randn(SHAPE)
    out = ttnn.min(to_device(device, x, dtype), dim=dim, keepdim=True)
    _check_fallback(out, _fallback_golden(x, dtype, lambda t, dim: t.amin(dim, keepdim=True), dim=dim), dtype)


def test_reduce_mx_hw_skipped(device):
    pytest.skip("sum / mean over H and W in one call is wrong on Quasar for bf16 as well (pre-existing, #62742)")


@pytest.mark.parametrize("dtype", TWO_DTYPES)
@pytest.mark.parametrize("dim", [-1, -2], ids=["W", "H"])
def test_reduce_mx_keepdim_false(device, dtype, dim):
    # The squeezing reshape takes bfloat16 / uint32 only, so the squeezed result comes from the bfloat16 route.
    torch.manual_seed(7)
    x = torch.randn(1, 1, 32, 64)
    out = ttnn.sum(to_device(device, x, dtype), dim=dim, keepdim=False)
    _check_fallback(out, _fallback_golden(x, dtype, torch.sum, dim=dim, keepdim=False), dtype)


@pytest.mark.parametrize("dtype", TWO_DTYPES)
def test_reduce_mx_full_reduce(device, dtype):
    # dim=None reduces every axis, N and C included. The result is a single value, so PCC (a correlation of one point)
    # says nothing: compare with allclose. The tolerance covers the MX output quantization (the device sums in
    # bfloat16, so the value can land one MX step from the float32 sum) and the bfloat16 accumulation.
    torch.manual_seed(8)
    x = torch.randn(1, 2, 32, 32)
    out = ttnn.sum(to_device(device, x, dtype))
    assert out.dtype == dtype, f"output dtype {out.dtype}, expected {dtype}"
    golden = _fallback_golden(x, dtype, lambda t: t.sum(dim=(0, 1, 2, 3), keepdim=True)).reshape(-1)
    actual = ttnn.to_torch(out).to(torch.float32).reshape(-1)[: golden.numel()]
    assert torch.isfinite(actual).all(), "non-finite value in the device output"
    torch.testing.assert_close(actual, golden.to(torch.float32), rtol=0.15, atol=0.1)


def test_reduce_mx_rejects_row_major_output(device, expect_error):
    # MX exists in TILE layout only: a ROW_MAJOR result would silently come back as BFLOAT16.
    x = to_device(device, torch.randn(1, 1, 32, 32), ttnn.mxint8)
    with expect_error(RuntimeError, "output_layout=ROW_MAJOR is not supported"):
        ttnn.sum(x, dim=-1, keepdim=True, output_layout=ttnn.ROW_MAJOR_LAYOUT)

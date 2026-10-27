# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.sum / mean / max over MX (OCP microscaling) inputs on Quasar (issue #62742).

The tiled H / W reduce kernels unpack MX to Float16_b, reduce on the FPU and pack the result back to the input's
MX format (the reductions have no output dtype argument). Not wired for MX: implicit tile padding and the N / C
axes (transpose / permute of MX tiles), both rejected by the op.

MXFP4 column (H) sums: on main the LLK moves an MxFp4 data operand to the 2x-packed src format while the bf16
scaler stays plain, a pairing the FPU refuses; #59597 keeps the plain path unless the scaler is MXFP4 too. Those
cases are skipped until it lands.

Known gaps (skipped below, see issue #62742): ttnn.min (the fused -MAX(-x) kernels copy a tile back out of a
scratch ring, and that copy returns zeros with an MX input even when the scratch ring is bf16); reductions over
both H and W in one call (wrong on Quasar for bf16 too); std / var (the welford op rejects Quasar for every dtype);
keepdim=False (the squeezing reshape is not wired for MX tiles, rejected by the op).
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
    if dtype == ttnn.mxfp4 and dim == -2 and op in ("sum", "mean"):
        pytest.skip("MXFP4 column SUM/AVG needs the plain-path fallback of #59597 (bf16 scaler with 2x data)")
    torch.manual_seed(0)
    x = torch.randn(SHAPE)
    tt_op, torch_op = _OPS[op]
    out = tt_op(to_device(device, x, dtype), dim=dim, keepdim=True)
    assert out.dtype == dtype
    golden = mx_round_trip(torch_op(mx_round_trip(x, dtype), dim), dtype)
    check(golden, ttnn.to_torch(out), pcc_floor(dtype))


def test_reduce_mx_rejects_nc_axis(device, expect_error):
    x = to_device(device, torch.randn(1, 2, 32, 32), ttnn.mxint8)
    with expect_error(RuntimeError, "H and W axes only"):
        ttnn.sum(x, dim=1, keepdim=True)


def test_reduce_mx_rejects_implicit_padding(device, expect_error):
    x = to_device(device, torch.randn(1, 1, 40, 32), ttnn.mxint8)
    with expect_error(RuntimeError, "tile-aligned H and W"):
        ttnn.sum(x, dim=-1, keepdim=True)


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


@pytest.mark.parametrize("dtype", ttnn.MX_DTYPES)
@pytest.mark.parametrize("dim", [-1, -2], ids=["W", "H"])
def test_min_mx(device, dtype, dim):
    pytest.skip(
        "ttnn.min on an MX input returns zeros: the fused -MAX(-x) kernels (reduce_w_neg / reduce_h_neg) copy a "
        "tile out of a scratch ring and that copy_tile is broken for MX inputs on Quasar (issue #62742)"
    )


def test_reduce_mx_hw_skipped(device):
    pytest.skip("sum / mean over H and W in one call is wrong on Quasar for bf16 as well (pre-existing, #62742)")


def test_reduce_mx_rejects_keepdim_false(device, expect_error):
    x = to_device(device, torch.randn(1, 1, 32, 32), ttnn.mxint8)
    with expect_error(RuntimeError, "keepdim=True"):
        ttnn.sum(x, dim=-1, keepdim=False)


def test_reduce_mx_rejects_full_reduce(device, expect_error):
    # dim=None reduces every axis, and the N / C axes are not supported for MX.
    x = to_device(device, torch.randn(1, 1, 32, 32), ttnn.mxint8)
    with expect_error(RuntimeError, "H and W axes only"):
        ttnn.sum(x)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.sum / mean / max over MX (OCP microscaling) inputs on Quasar (issue #62742).

The tiled H / W reduce kernels unpack MX to Float16_b, reduce on the FPU and pack the result back to the input's
MX format (the reductions have no output dtype argument). Not wired for MX: implicit tile padding and the N / C
axes (transpose / permute of MX tiles), both rejected by the op.

MXFP4 column (H) sums: on main the LLK moves an MxFp4 data operand to the 2x-packed src format while the bf16
scaler stays plain, a pairing the FPU refuses; #59597 keeps the plain path unless the scaler is MXFP4 too. Those
cases are skipped until it lands.
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

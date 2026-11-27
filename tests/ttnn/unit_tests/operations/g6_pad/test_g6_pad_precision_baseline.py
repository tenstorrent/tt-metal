# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for g6_pad (a pure data move: the expected result is bit equality).

Measures PCC (comp_pcc), max abs error (comp_allclose), mean abs error, relative RMS (RMS error / std
of the reference, as eval.metrics.check_output), the max ULP distance (integer distance of the bit
patterns) and the got/true ratio spread (scale-bug detector) on production shapes of both regimes and
both float dtypes. Gate: ULP 0 everywhere (contract), which implies PCC 1 and zero error.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.g6_pad import g6_pad

from .test_g6_pad import to_torch_pad

CASES = [
    pytest.param((1, 1, 2, 62080), ((0, 0), (0, 0), (0, 30), (0, 0)), 0.0, ttnn.bfloat16, id="p2_ext_bf16"),
    pytest.param((1, 3844, 1536), ((0, 0), (0, 28), (0, 0)), 0.1, ttnn.bfloat16, id="c247_ext_bf16_inexact_c"),
    pytest.param((1, 1, 196, 196), ((0, 60), (0, 60)), -1.7014118346046923e38, ttnn.bfloat16, id="c65_new_bf16"),
    pytest.param((5, 12, 128, 128), ((0, 3), (0, 0), (0, 0), (0, 0)), 0, ttnn.float32, id="p1_new_fp32"),
]


def _ulp(a, b):
    view = torch.int16 if a.element_size() == 2 else torch.int32
    ia = a.contiguous().view(view).to(torch.int64)
    ib = b.contiguous().view(view).to(torch.int64)
    return int((ia - ib).abs().max())


@pytest.mark.parametrize("shape, padding, value, dtype", CASES)
def test_g6_pad_precision_baseline(device, shape, padding, value, dtype):
    torch.manual_seed(0)
    tdtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    x = (torch.randn(shape) * 100).to(tdtype)
    expected = torch.nn.functional.pad(x, to_torch_pad(padding), mode="constant", value=value)
    x_tt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    x_tt = ttnn.fill_implicit_tile_padding(x_tt, 1000.0)
    got = ttnn.to_torch(g6_pad(x_tt, padding, value=value))

    ulp = _ulp(got, expected)
    e64, g64 = expected.double(), got.double()
    pcc_ok, pcc = comp_pcc(e64, g64, 0.9999)
    _, allclose_msg = comp_allclose(e64, g64)
    err = g64 - e64
    mean_abs = float(err.abs().mean())
    rel_rms = float(err.pow(2).mean().sqrt() / e64.std())
    nz = (e64 != 0) & torch.isfinite(e64)
    r = g64[nz][: 1 << 20]
    r = r / e64[nz][: 1 << 20]
    q = torch.quantile(r, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    print(
        f"\nBASELINE {list(shape)} {dtype}: pcc={pcc} [{allclose_msg}] mean_abs={mean_abs:.3e} "
        f"rel_rms={rel_rms:.3e} max_ulp={ulp} ratio p5/p50/p95={q[0]:.6f}/{q[1]:.6f}/{q[2]:.6f}"
    )
    assert ulp == 0, f"bit mismatch: max ULP distance {ulp}"
    assert pcc_ok and rel_rms == 0.0

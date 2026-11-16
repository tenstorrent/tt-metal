# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision baseline for wq_b_linear.

The op accepts exactly one shape, so the sweep is over input distributions and seeds.
Reference: fp64 matmul of x and the device-quantized (bfloat8_b) W.
"""

import pytest
import torch
import ttnn

from eval.golden_tests.wq_b_linear.helpers import TOLERANCES, WEIGHT_SCALE
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.wq_b_linear import wq_b_linear

X_SHAPE = (1, 1, 160, 1536)
W_SHAPE = (1536, 4096)
PCC = TOLERANCES[(ttnn.bfloat8_b, False)]
RMS = 0.02


def _inputs(case, seed):
    torch.manual_seed(seed)
    x = torch.randn(X_SHAPE)
    w = torch.randn(W_SHAPE) * WEIGHT_SCALE
    if case == "x_scaled":
        x = x * 8.0
    elif case == "x_offset":
        x = x + 0.5
    elif case == "uniform":
        x = torch.rand(X_SHAPE) * 2 - 1
        w = (torch.rand(W_SHAPE) * 2 - 1) * WEIGHT_SCALE
    return x.to(torch.bfloat16), w.to(torch.bfloat16)


def _ulp_bf16(actual, expected):
    exp_bits = expected.to(torch.bfloat16).float().abs().clamp(min=torch.finfo(torch.bfloat16).tiny)
    ulp = torch.pow(2.0, torch.floor(torch.log2(exp_bits)) - 7)
    return ((actual - expected).abs() / ulp).flatten()


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("case", ["standard", "x_scaled", "x_offset", "uniform"])
def test_precision_baseline(device, case, seed):
    x, w = _inputs(case, seed)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    expected = torch.matmul(x.double(), ttnn.to_torch(tw).double())
    actual = ttnn.to_torch(wq_b_linear(tx, tw)).double()

    _, pcc = comp_pcc(expected, actual, PCC)
    _, abs_msg = comp_allclose(expected, actual)
    diff = actual - expected
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    rel_rms = (diff.pow(2).mean().sqrt() / expected.std()).item()
    ulp = _ulp_bf16(actual, expected)
    ulp_p99 = torch.quantile(ulp[:1_000_000].float(), 0.99).item()

    mask = expected.abs() > 1e-3 * expected.abs().max()
    ratio = (actual[mask] / expected[mask]).float()
    r_med = ratio.median().item()
    r_p5, r_p95 = torch.quantile(ratio[:1_000_000], torch.tensor([0.05, 0.95])).tolist()

    print(
        f"\nPRECISION case={case} seed={seed} pcc={pcc:.6f} max_abs={max_abs:.5f} mean_abs={mean_abs:.5f} "
        f"rel_rms={rel_rms:.5f} ulp_p99={ulp_p99:.1f} ratio_med={r_med:.5f} ratio_p5={r_p5:.5f} "
        f"ratio_p95={r_p95:.5f} ({abs_msg})"
    )
    assert pcc >= PCC
    assert rel_rms <= RMS

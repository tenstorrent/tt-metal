# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision baseline for wq_b_linear.

The op accepts exactly one shape (x [1,1,160,1536] bf16, W [1536,4096] bf8_b), so the
baseline sweeps input *profiles* (distribution / seed) instead of shapes:
  * randn_s0 / randn_s7 — the golden distribution (x ~ N(0,1), W ~ 0.02·N(0,1)), two seeds
  * pos_mean           — x ~ U(0,1), W ~ 0.02·N(0.5,1): non-zero mean, long same-sign K sums
  * wide_range         — x ~ 8·N(0,1), W ~ 0.02·N(0,1): 8× larger output magnitude

Metrics: PCC (comp_pcc), max/mean abs error, relative RMS (RMS err / ref std, as in
eval/metrics.check_output), ULP p99 (eval/metrics.compute_metrics), and the got/true ratio
spread (scale-bug detector: a tight cluster at a non-1.0 constant = scale bug).
"""

import re

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from eval.metrics import DEFAULT_TOLERANCES, compute_metrics
from eval.golden_tests.wq_b_linear.helpers import TOLERANCES, WEIGHT_SCALE
from ttnn.operations.wq_b_linear import wq_b_linear

X_SHAPE = (1, 1, 160, 1536)
W_SHAPE = (1536, 4096)
# Golden-distribution profiles hold the op's own gate. pos_mean is format-limited: rounding the
# exact result to bf16 alone gives PCC 0.99968 / rel-RMS 0.025 (output mean ~8x its std), and 16-bit
# DEST accumulation of same-sign sums adds to that (measured 0.9954 / 0.097), so it is gated at the
# shared (bfloat8_b, fp32_dest_acc_en=False) default from eval/metrics.DEFAULT_TOLERANCES.
GATES = {
    "randn_s0": (TOLERANCES[(ttnn.bfloat8_b, False)], 0.03),
    "randn_s7": (TOLERANCES[(ttnn.bfloat8_b, False)], 0.03),
    "wide_range": (TOLERANCES[(ttnn.bfloat8_b, False)], 0.03),
    "pos_mean": DEFAULT_TOLERANCES[(ttnn.bfloat8_b, False)],
}
# |global scale - 1| guard: 16-bit DEST drifts the scale up with block_k_tiles (1.007 at 8,
# 1.013 at 16); 0.02 allows the whole PCC-legal block_k_tiles range and still catches a real scale bug.
SCALE_TOL = 0.02


def _profile(name):
    if name == "randn_s0":
        torch.manual_seed(0)
        return torch.randn(X_SHAPE), torch.randn(W_SHAPE) * WEIGHT_SCALE
    if name == "randn_s7":
        torch.manual_seed(7)
        return torch.randn(X_SHAPE), torch.randn(W_SHAPE) * WEIGHT_SCALE
    if name == "pos_mean":
        torch.manual_seed(1)
        return torch.rand(X_SHAPE), (torch.randn(W_SHAPE) + 0.5) * WEIGHT_SCALE
    if name == "wide_range":
        torch.manual_seed(2)
        return torch.randn(X_SHAPE) * 8.0, torch.randn(W_SHAPE) * WEIGHT_SCALE
    raise ValueError(name)


@pytest.mark.parametrize("profile", ["randn_s0", "randn_s7", "pos_mean", "wide_range"])
def test_wq_b_linear_precision_baseline(device, profile):
    x, w = _profile(profile)
    x = x.to(torch.bfloat16)
    w = w.to(torch.bfloat16)
    tx = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tw = ttnn.from_torch(
        w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    # Reference on the device-quantized weight: measures the op, not the bf8_b format.
    w_q = ttnn.to_torch(tw).to(torch.float64)
    expected = torch.matmul(x.to(torch.float64), w_q)

    out = wq_b_linear(tx, tw)
    got = ttnn.to_torch(out).to(torch.float64)

    pcc_min, rms_max = GATES[profile]
    _, pcc = comp_pcc(expected, got, pcc_min)
    _, allclose_msg = comp_allclose(expected, got)
    max_abs = float(re.search(r"Max ATOL Delta: ([0-9.eE+-]+)", allclose_msg).group(1))
    mean_abs = (got - expected).abs().mean().item()
    m = compute_metrics(out, expected.to(torch.float32))

    nz = expected.abs() > 1e-3 * expected.abs().max()
    r = (got[nz] / expected[nz]).flatten()
    r_med = r.median().item()
    slope = ((got * expected).sum() / (expected * expected).sum()).item()  # least-squares global scale
    r_p5, r_p95 = torch.quantile(r[: 1 << 20].float(), torch.tensor([0.05, 0.95])).tolist()

    print(
        f"\nPRECISION {profile}: pcc={pcc:.6f} max_abs={max_abs:.5f} mean_abs={mean_abs:.5f} "
        f"rel_rms={m.rms:.5f} ulp_p99={m.ulp_p99} slope={slope:.5f} ratio_median={r_med:.5f} ratio_p5={r_p5:.4f} ratio_p95={r_p95:.4f}"
    )
    assert pcc >= pcc_min, f"PCC {pcc:.6f} < {pcc_min}"
    assert m.rms <= rms_max, f"relative RMS {m.rms:.5f} > {rms_max}"
    assert abs(slope - 1.0) < SCALE_TOL, f"global scale {slope:.5f} (scale bug?)"

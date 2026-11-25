# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Precision baseline for groupnorm_sc_N_1_HW_C_b (Phase 0: bf16, DRAM interleaved, gamma+beta bf16 RM).

Per shape it records PCC (comp_pcc), max abs error (comp_allclose), mean abs error, relative RMS error
(RMS error / std(reference), as eval.metrics.check_output computes it) and the got/true ratio spread
(median and p5..p95 of actual/expected over finite, non-tiny reference elements). A tight ratio cluster
around a constant != 1.0 is the signature of a uniform scale/structural bug; a broad spread centred on
1.0 is ordinary rounding noise.

Gates: PCC and relative RMS together against helpers.TOLERANCES[bf16] (the perf-case rows against
feature_spec.PERF_TOLERANCE, the bar their loose cases are held to).

The offset-heavy rows (per-group mean 5 / 10, std 1) feed the same distribution as the golden
test_regression.py::test_sharded_offset_heavy through the DRAM path: the math is the same kernel, so
these rows show what Refinement 1 / 2 inherit from the one-pass E[x^2] - mean^2 statistics.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc

from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.feature_spec import PERF_TOLERANCE
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import (
    TOLERANCES,
    create_ttnn_input_tensor,
    pytorch_groupnorm_sc_N_1_HW_C,
)
from ttnn.operations.groupnorm_sc_N_1_HW_C_b import groupnorm_sc_N_1_HW_C_b

BF16_TOL = TOLERANCES[ttnn.bfloat16]

# (shape, num_groups, layout, eps, offset, tolerance)
CASES = [
    pytest.param((1, 1, 32, 32), 1, ttnn.TILE_LAYOUT, 1e-5, 0.0, BF16_TOL, id="small_1tile"),
    pytest.param((2, 1, 512, 256), 8, ttnn.TILE_LAYOUT, 1e-5, 0.0, BF16_TOL, id="batched_cg32"),
    pytest.param((1, 1, 1024, 640), 32, ttnn.ROW_MAJOR_LAYOUT, 1e-5, 0.0, BF16_TOL, id="medium_cg20_rm"),
    pytest.param((1, 1, 16384, 1152), 32, ttnn.TILE_LAYOUT, 1e-5, 0.0, PERF_TOLERANCE, id="large_refiner_cg36"),
    pytest.param((1, 1, 65536, 128), 32, ttnn.TILE_LAYOUT, 1e-6, 0.0, PERF_TOLERANCE, id="large_vae_cg4"),
    pytest.param((1, 1, 1024, 1280), 32, ttnn.TILE_LAYOUT, 1e-5, 5.0, BF16_TOL, id="offset_mean5"),
    pytest.param(
        (1, 1, 1024, 1280),
        32,
        ttnn.TILE_LAYOUT,
        1e-5,
        10.0,
        BF16_TOL,
        id="offset_mean10",
        # Phase 0 measured PCC 0.99972, rel-RMS 0.0273 (> 0.02): one-pass E[x^2] - mean^2 cancels at mean/std = 10,
        # amplifying the TF32 truncation of x^2 / the channel sums in SrcA. Refinement 1 shifts both moments by a
        # per-group pilot sample k (sum (x-k)^2 - (sum (x-k))^2 / n), so the cancellation scales with var.
    ),
]


def _metrics(actual, expected):
    a = actual.to(torch.float32).flatten()
    e = expected.to(torch.float32).flatten()
    _, pcc = comp_pcc(e, a)
    _, allclose_msg = comp_allclose(e, a)
    max_abs = float(allclose_msg.split("Max ATOL Delta: ")[1].split(",")[0])
    diff = a - e
    mean_abs = diff.abs().mean().item()
    rel_rms = (diff.pow(2).mean().sqrt() / e.std()).item()
    # Ratio over elements whose reference is not near zero (|e| > 0.1 std), so bf16 output rounding of
    # near-zero values does not dominate the spread. torch.quantile caps its input size: sample a prefix.
    keep = torch.isfinite(e) & (e.abs() > 0.1 * e.std())
    r = (a[keep] / e[keep])[: 1 << 22].to(torch.float64)
    q = torch.quantile(r, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    return dict(
        pcc=float(pcc),
        max_abs=max_abs,
        mean_abs=mean_abs,
        rel_rms=rel_rms,
        r_p5=q[0].item(),
        r_med=q[1].item(),
        r_p95=q[2].item(),
    )


@pytest.mark.parametrize("shape, num_groups, layout, eps, offset, tolerance", CASES)
def test_precision_baseline(device, shape, num_groups, layout, eps, offset, tolerance):
    torch.manual_seed(0)
    C = shape[-1]
    x = (torch.randn(shape, dtype=torch.float32) + offset).to(torch.bfloat16)
    gamma = torch.randn((1, 1, 1, C), dtype=torch.float32).to(torch.bfloat16)
    beta = torch.randn((1, 1, 1, C), dtype=torch.float32).to(torch.bfloat16)
    expected = pytorch_groupnorm_sc_N_1_HW_C(x, num_groups, gamma=gamma, beta=beta, eps=eps)

    tt_x = create_ttnn_input_tensor(x, device, dtype=ttnn.bfloat16, layout=layout)
    tt_g = create_ttnn_input_tensor(gamma, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_b = create_ttnn_input_tensor(beta, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    out = groupnorm_sc_N_1_HW_C_b(tt_x, num_groups, gamma=tt_g, beta=tt_b, eps=eps)
    actual = ttnn.to_torch(out)

    m = _metrics(actual, expected)
    print(
        f"\nPRECISION shape={shape} G={num_groups} layout={layout} offset={offset} "
        f"pcc={m['pcc']:.6f} max_abs={m['max_abs']:.4f} mean_abs={m['mean_abs']:.5f} "
        f"rel_rms={m['rel_rms']:.5f} ratio[p5,med,p95]=[{m['r_p5']:.4f},{m['r_med']:.4f},{m['r_p95']:.4f}]"
    )
    pcc_min, rms_max = tolerance
    assert m["pcc"] >= pcc_min, f"PCC {m['pcc']:.6f} < {pcc_min}"
    assert m["rel_rms"] <= rms_max, f"relative RMS {m['rel_rms']:.5f} > {rms_max}"

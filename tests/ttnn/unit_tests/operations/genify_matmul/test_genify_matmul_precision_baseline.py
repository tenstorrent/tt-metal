# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision baseline for genify_matmul (Phase 0 cell: bf16 x bf16 -> bf16, HiFi4, fp32 DEST, interleaved).

Measures PCC (comp_pcc), max / mean abs error (comp_allclose + torch), relative RMS (RMS error over
the reference's std, as eval.metrics.check_output), ULP p99 (eval.metrics) and the got/true ratio
spread (scale-bug detector: a tight cluster at a non-1.0 constant means a structural bug, a broad
spread centred on 1.0 is rounding noise). Gates on PCC and relative RMS together at
helpers.TOLERANCES[(bf16, True)]; the other numbers are printed for verification_report.md.

Shapes cover the R1 sub-regimes: single tile; 1-D N split (M = 1); 2-D grid with one resident K-block;
deep K with a multi-K-block packer-L1 walk; a large 2-D shape with the S3 block shrink.
"""

import math

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc

from eval.golden_tests.genify_matmul.helpers import (
    TOLERANCES,
    create_ttnn_input_tensor,
    pytorch_genify_matmul,
    quantize,
)
from eval.metrics import compute_metrics_torch
from ttnn.operations.genify_matmul import genify_matmul

SHAPES = [
    ((1, 1, 32, 32), (32, 32)),  # single tile
    ((1, 2048), (2048, 1536)),  # decode M = 1, 1-D grid
    ((1, 384, 1536), (1536, 1024)),  # prefill / DiT, one resident K-block
    ((1, 64, 16384), (16384, 512)),  # deep K, multi K-block walk (packer L1 accumulation)
    ((1, 2048, 4096), (4096, 2048)),  # large 2-D, K-walk
]
IDS = [f"A{'x'.join(map(str, a))}_B{'x'.join(map(str, b))}" for a, b in SHAPES]


def _ratio_spread(actual, expected):
    """got/true over finite elements whose reference is not tiny (|ref| > 10% of its std)."""
    mask = expected.abs() > 0.1 * expected.std()
    r = (actual[mask] / expected[mask]).double()
    r = r[torch.isfinite(r)]
    q = torch.quantile(r[: 1 << 24], torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    return q[1].item(), q[0].item(), q[2].item()


@pytest.mark.parametrize("a_shape, b_shape", SHAPES, ids=IDS)
def test_precision_baseline(a_shape, b_shape, device):
    torch.manual_seed(1234)
    A = torch.randn(a_shape)
    B = torch.randn(b_shape) / math.sqrt(b_shape[-2])
    ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tb = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    expected = pytorch_genify_matmul(quantize(A, ttnn.bfloat16), quantize(B, ttnn.bfloat16))

    actual = ttnn.to_torch(genify_matmul(ta, tb)).float()
    assert list(actual.shape) == list(expected.shape)

    _, pcc = comp_pcc(expected, actual)
    _, allclose_msg = comp_allclose(expected, actual)
    diff = (actual - expected).abs()
    max_abs, mean_abs = diff.max().item(), diff.mean().item()
    rel_rms = math.sqrt(diff.pow(2).mean().item()) / expected.std().item()
    ulp_p99 = compute_metrics_torch(actual, expected, readback_dtype=torch.bfloat16).ulp_p99
    r_med, r_p5, r_p95 = _ratio_spread(actual, expected)

    print(
        f"\nPRECISION {a_shape} x {b_shape}: pcc={pcc:.7f} max_abs={max_abs:.5f} mean_abs={mean_abs:.6f} "
        f"rel_rms={rel_rms:.5f} ulp_p99={ulp_p99:.1f} ratio median={r_med:.5f} p5={r_p5:.5f} p95={r_p95:.5f} "
        f"[{allclose_msg}]"
    )

    pcc_min, rms_max = TOLERANCES[(ttnn.bfloat16, True)]
    assert pcc >= pcc_min, f"PCC {pcc} < {pcc_min}"
    assert rel_rms <= rms_max, f"relative RMS {rel_rms} > {rms_max}"
    # Scale-bug detector: the median got/true ratio must sit at 1 (bf16 output rounding is <= 2^-8).
    assert abs(r_med - 1.0) < 4e-3, f"median got/true ratio {r_med} != 1 (uniform scale error)"

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for groupnorm_sc_N_1_HW_C (Phase 0: bfloat16 only).

Measures PCC, max/mean abs error, and relative RMS error across 4 shapes
spanning single-tile, medium, batched, and larger streaming-stat slabs.
Results recorded in verification_report.md. Tolerances are baselines, not
targets — they document Phase-0 numerics for future refinement comparison.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

PCC_THRESHOLD = 0.995  # bf16 baseline
RMS_THRESHOLD = 0.02  # relative RMS vs reference stddev


def torch_groupnorm(x, num_groups, gamma, beta, eps=1e-5):
    N, _, HW, C = x.shape
    x_nchw = x.to(torch.float32).squeeze(1).permute(0, 2, 1)
    w = gamma.to(torch.float32).reshape(C)
    b = beta.to(torch.float32).reshape(C)
    y = torch.nn.functional.group_norm(x_nchw, num_groups, weight=w, bias=b, eps=eps)
    return y.permute(0, 2, 1).unsqueeze(1)


SHAPES = [
    pytest.param((1, 1, 32, 32), 1, id="small_32x32_g1"),
    pytest.param((1, 1, 128, 128), 4, id="medium_128x128_g4"),
    pytest.param((2, 1, 64, 128), 4, id="batched_2x64x128_g4"),
    pytest.param((1, 1, 512, 256), 8, id="large_512x256_g8"),
]


@pytest.mark.parametrize("shape, num_groups", SHAPES)
def test_precision_baseline(device, shape, num_groups):
    torch.manual_seed(1234)
    N, _, HW, C = shape
    x = torch.randn(shape, dtype=torch.bfloat16)
    gamma = torch.randn(1, 1, 1, C, dtype=torch.bfloat16)
    beta = torch.randn(1, 1, 1, C, dtype=torch.bfloat16)

    expected = torch_groupnorm(x, num_groups, gamma, beta)

    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_g = ttnn.from_torch(gamma, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_b = ttnn.from_torch(beta, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    result = ttnn.to_torch(groupnorm_sc_N_1_HW_C(tt_x, num_groups, gamma=tt_g, beta=tt_b)).to(torch.float32)

    abs_err = (result - expected).abs()
    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    rel_rms = (abs_err.pow(2).mean().sqrt() / expected.std()).item()

    allclose_pass, allclose_str = comp_allclose(expected, result, rtol=0.05, atol=0.06)
    pcc = torch.corrcoef(torch.stack([expected.flatten(), result.flatten()]))[0, 1].item()
    print(
        f"\nPRECISION shape={shape} G={num_groups}: pcc={pcc:.6f} max_abs={max_abs:.5f} "
        f"mean_abs={mean_abs:.5f} rel_rms={rel_rms:.5f} | {allclose_str}"
    )

    assert_with_pcc(expected, result, pcc=PCC_THRESHOLD)
    assert rel_rms < RMS_THRESHOLD, f"relative RMS {rel_rms:.5f} exceeds baseline {RMS_THRESHOLD}"

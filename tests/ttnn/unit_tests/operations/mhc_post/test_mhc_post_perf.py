# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf probe shapes for mhc_post (run under scripts/run_safe_pytest.sh --profile)."""

import pytest
import torch
import ttnn

from ttnn.operations.mhc_post import mhc_post

N = 4


@pytest.mark.parametrize("T,C", [(160, 5120), (640, 7168), (1024, 1792)])
def test_mhc_post_perf_plain_fp32(device, T, C):
    torch.manual_seed(0)
    dev = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    f, x = dev(torch.randn(1, 1, T, C)), dev(torch.randn(1, 1, T, N * C))
    post, comb = dev(torch.rand(1, 1, T, N)), dev(torch.rand(1, 1, T, N * N))
    for _ in range(2):
        mhc_post(f, x, post, comb)
    ttnn.synchronize_device(device)


@pytest.mark.parametrize("mode", ["plain", "stats", "collapse"])
@pytest.mark.parametrize("T,C", [(160, 5120), (1280, 5120)])
def test_mhc_post_perf_ds41(device, T, C, mode):
    """DeepSeek-V4.1's call forms: bf16 streams, packed mixes, HiFi2 (R2 fused-gather for stats / collapse)."""
    from eval.golden_tests.mhc_post.feature_spec import _case
    from eval.golden_tests.mhc_post.helpers import call_mhc_post, make_case

    case = make_case(
        _case((1, 1, T, N * C)), mode=mode, coef_source="packed", dtype=ttnn.bfloat16, sublayer_dtype=ttnn.bfloat16
    )
    for _ in range(2):
        call_mhc_post(
            case,
            mode=mode,
            coef_source="packed",
            dtype=ttnn.bfloat16,
            sublayer_dtype=ttnn.bfloat16,
            device=device,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            op=mhc_post,
        )
    ttnn.synchronize_device(device)


@pytest.mark.parametrize("iters", [1, 20])
def test_mhc_post_perf_hc_iters(device, iters):
    """Ablation: the hc step's Sinkhorn trip count (collapse, T160 C5120)."""
    from eval.golden_tests.mhc_post.feature_spec import _case
    from eval.golden_tests.mhc_post.helpers import call_mhc_post, make_case

    case = make_case(
        _case((1, 1, 160, N * 5120)),
        mode="collapse",
        coef_source="packed",
        dtype=ttnn.bfloat16,
        sublayer_dtype=ttnn.bfloat16,
    )
    case.hc["sinkhorn_iters"] = iters
    for _ in range(2):
        call_mhc_post(
            case,
            mode="collapse",
            coef_source="packed",
            dtype=ttnn.bfloat16,
            sublayer_dtype=ttnn.bfloat16,
            device=device,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            op=mhc_post,
        )
    ttnn.synchronize_device(device)


def test_mhc_post_perf_collapse_t160(device):
    """One parameter-free collapse call (T160 C5120) for zone profiling (MHC_ABLATE=zones)."""
    test_mhc_post_perf_hc_iters(device, 20)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression test for tenstorrent/tt-metal#57965 ($3,000 bounty).

Bug: Under FLOAT32 input with fp32_dest_acc_en=True, pre_all_gather writes the true row
sum to column 0, but leaves intermediate partial sums in column 16 (and 48 for LayerNorm)
due to the accurate SFPU horizontal reduction butterfly.

Previously, post_all_gather's reader prepared the reduce scaler using
valid_reduce_dim_elements_in_tile=32 (default), summing column 16 into the stats and
inflating E[x^2] by ~3.2% (causing error ~0.112 vs PyTorch float64 instead of <=0.0059).

Fix: Passing valid_reduce_dim_elements_in_tile=1 in the post reader zero-masks columns 1-31,
ensuring only column 0 is reduced across devices.
"""

import pytest
import torch
import ttnn

from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.nightly.unit_tests.operations.fused.utility_functions import (
    ttnn_layer_norm_pre_all_gather,
    ttnn_layer_norm_post_all_gather,
    ttnn_rms_norm_pre_all_gather,
    ttnn_rms_norm_post_all_gather,
)

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 544),  # 17 tiles wide (exact reproduction shape from #57965)
        (1, 1, 32, 32),   # 1 tile wide
        (1, 1, 32, 64),   # 2 tiles wide
        (1, 1, 64, 544),  # Multi-row height
    ],
)
@pytest.mark.parametrize("offset", [0.0, 1.0])
def test_rmsnorm_distributed_fp32_stats_columns(device, shape, offset):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32) + offset
    t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )

    # 1. Run pre_all_gather: gather is a no-op on a single device
    stats = ttnn_rms_norm_pre_all_gather(t, compute_kernel_config=cfg, dtype=ttnn.float32)
    s = ttnn.to_torch(stats)

    # Verify that SFPU accurate reduction left non-zero data in col 16
    non_zero_cols = [c for c in range(s.shape[-1]) if s[..., c].abs().max().item() > 0]
    assert 0 in non_zero_cols, "Column 0 must contain row sum"

    # 2. Run post_all_gather: verify that column 16 does NOT pollute the reduction
    eps = 1e-5
    y_tt = ttnn_rms_norm_post_all_gather(t, stats, epsilon=eps, compute_kernel_config=cfg)
    y = ttnn.to_torch(y_tt)

    # PyTorch reference
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    max_abs_err = (y - ref).abs().max().item()
    logger.info(f"RMSNorm shape={shape} offset={offset}: max_abs_err={max_abs_err:.6f}")

    # Prior to fix, max_abs_err was ~0.112 (offset=0) / ~0.077 (offset=1)
    # With fix, error must be <= 0.0059
    assert max_abs_err <= 0.0059, f"Max absolute error {max_abs_err} exceeds 0.0059"


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 544),
        (1, 1, 32, 64),
    ],
)
@pytest.mark.parametrize("offset", [0.0, 1.0])
def test_layernorm_distributed_fp32_stats_columns(device, shape, offset):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32) + offset
    t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )

    stats = ttnn_layer_norm_pre_all_gather(t, compute_kernel_config=cfg, dtype=ttnn.float32)
    s = ttnn.to_torch(stats)

    # LayerNorm stats are 2 tiles wide (64 cols): sum(x^2) at col 0, sum(x) at col 32
    non_zero_cols = [c for c in range(s.shape[-1]) if s[..., c].abs().max().item() > 0]
    assert 0 in non_zero_cols and 32 in non_zero_cols

    eps = 1e-5
    y_tt = ttnn_layer_norm_post_all_gather(t, stats, epsilon=eps, compute_kernel_config=cfg)
    y = ttnn.to_torch(y_tt)

    ref = torch.nn.functional.layer_norm(x, (shape[-1],), eps=eps)
    max_abs_err = (y - ref).abs().max().item()
    logger.info(f"LayerNorm shape={shape} offset={offset}: max_abs_err={max_abs_err:.6f}")

    # Prior to fix, max_abs_err was ~0.117 (offset=0) / ~0.081 (offset=1)
    # With fix, error must be <= 0.0055
    assert max_abs_err <= 0.0055, f"Max absolute error {max_abs_err} exceeds 0.0055"

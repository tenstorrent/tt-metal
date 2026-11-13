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

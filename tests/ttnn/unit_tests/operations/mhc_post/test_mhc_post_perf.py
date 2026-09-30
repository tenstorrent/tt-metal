# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf-measurement shapes for mhc_post (run under scripts/run_safe_pytest.sh --profile)."""

import pytest
import torch
import ttnn

import ttnn.operations.mhc_post.mhc_post_program_descriptor as pd
from ttnn.operations.mhc_post import mhc_post

N = 4


@pytest.mark.parametrize("max_block", [None])
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("T, C", [(640, 7168), (640, 1792), (1280, 4096)])
def test_mhc_post_perf(device, T, C, dtype, max_block, monkeypatch):
    monkeypatch.setattr(pd, "MAX_BLOCK_COL_TILES", max_block)
    torch.manual_seed(0)
    f = torch.randn(1, 1, T, C)
    x = torch.randn(1, 1, T, N * C)
    post = torch.rand(1, 1, T, N) * 2
    comb = torch.rand(1, 1, T, N * N)

    def dev(t, dt):
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)

    args = (dev(f, dtype), dev(x, dtype), dev(post, ttnn.float32), dev(comb, ttnn.float32))
    if dtype == ttnn.bfloat16:
        f, x = f.bfloat16().float(), x.bfloat16().float()
    out = mhc_post(*args)
    ref = post.reshape(-1, N, 1) * f.reshape(-1, 1, C) + torch.einsum(
        "tij,tic->tjc", comb.reshape(-1, N, N), x.reshape(-1, N, C)
    )
    got = ttnn.to_torch(out).float().reshape(-1, N, C)
    tol = 1e-4 if dtype == ttnn.float32 else 2e-2
    assert torch.allclose(got, ref, rtol=tol, atol=tol)

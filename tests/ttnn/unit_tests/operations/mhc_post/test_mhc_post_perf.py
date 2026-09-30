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
@pytest.mark.parametrize("T, C", [(640, 7168), (640, 1792)])
def test_mhc_post_perf(device, T, C, max_block, monkeypatch):
    monkeypatch.setattr(pd, "MAX_BLOCK_COL_TILES", max_block)
    torch.manual_seed(0)
    f = torch.randn(1, 1, T, C)
    x = torch.randn(1, 1, T, N * C)
    post = torch.rand(1, 1, T, N) * 2
    comb = torch.rand(1, 1, T, N * N)

    def dev(t):
        return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    args = (dev(f), dev(x), dev(post), dev(comb))
    out = mhc_post(*args)
    ref = post.reshape(-1, N, 1) * f.reshape(-1, 1, C) + torch.einsum(
        "tij,tic->tjc", comb.reshape(-1, N, N), x.reshape(-1, N, C)
    )
    got = ttnn.to_torch(out).reshape(-1, N, C)
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-4)

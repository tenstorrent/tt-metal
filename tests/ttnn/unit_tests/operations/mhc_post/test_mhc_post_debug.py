# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic debug tests for mhc_post — DO NOT DELETE (documents the debugging process).

Each case isolates one operand path of the mix chain so the expected output is exact in fp32:
  f_path    : post = 1, comb = 0, X = 0           -> X'_j = F           (CopyTile(F) precision)
  post_path : post random, comb = 0, F = 1, X = 0 -> X'_j = post_j      (UnaryBcast<Col> precision)
  x_path    : post = 0, comb = I, F = 0           -> X'_j = X_j         (CopyTile(X) precision)
"""
import pytest
import torch
import ttnn

from ttnn.operations.mhc_post import mhc_post

N = 4


def _dev(t, device):
    return ttnn.from_torch(
        t.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


@pytest.mark.parametrize("case", ["f_path", "post_path", "x_path", "mul_path"])
def test_paths(case, device):
    torch.manual_seed(0)
    T, C = 32, 64
    f = torch.randn(T, C)
    x = torch.randn(T, N * C)
    post = torch.rand(T, N) + 0.5
    comb = torch.eye(N).expand(T, N, N).reshape(T, N * N).clone()
    if case == "f_path":
        post = torch.ones(T, N)
        comb.zero_()
        x.zero_()
        exp = f.repeat(1, N)
    elif case == "post_path":
        f = torch.ones(T, C)
        comb.zero_()
        x.zero_()
        exp = post.repeat_interleave(C, dim=1)
    elif case == "x_path":
        post.zero_()
        f.zero_()
        exp = x.clone()
    else:  # mul_path: post random, F random, comb 0
        comb.zero_()
        x.zero_()
        exp = (post.double().repeat_interleave(C, dim=1) * f.double().repeat(1, N)).float()
    out = ttnn.to_torch(mhc_post(_dev(f, device), _dev(x, device), _dev(post, device), _dev(comb, device)))
    diff = (out.double() - exp.double()).abs()
    rel = (diff / exp.double().abs().clamp_min(1e-30)).max().item()
    print(f"{case}: max abs diff {diff.max().item():.3g}, max rel {rel:.3g}")
    assert rel < 1e-6, f"{case}: max rel {rel:.3g}"

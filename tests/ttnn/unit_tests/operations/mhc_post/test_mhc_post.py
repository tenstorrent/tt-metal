# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for mhc_post (Phase 0: plain mode, float32 streams / sublayer, separate coefficients).

Immutable spec — the implementer must not modify this file.

    X'_j = post_j * F + sum_i comb[i][j] * X_i      (comb applied transposed)

F (..., T, C); X (..., T, n*C) with stream i in columns [i*C, (i+1)*C);
post (..., T, n) and comb (..., T, n*n) float32, comb[i][j] at column i*n + j.
"""

import pytest
import torch
import ttnn

from ttnn.operations.mhc_post import mhc_post

N = 4
PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}


def torch_mhc_post(f, x, post, comb):
    n = post.shape[-1]
    lead, C = tuple(f.shape[:-1]), f.shape[-1]
    f3 = f.double().reshape(-1, 1, C)
    x3 = x.double().reshape(-1, n, C)
    p3 = post.double().reshape(-1, n, 1)
    m = comb.double().reshape(-1, n, n)
    out = p3 * f3 + torch.einsum("tij,tic->tjc", m, x3)
    return out.reshape(*lead, n * C)


def make_inputs(lead, T, C):
    torch.manual_seed(42)
    f = torch.randn(*lead, T, C, dtype=torch.float32)
    x = torch.randn(*lead, T, N * C, dtype=torch.float32)
    post = 2.0 * torch.sigmoid(torch.randn(*lead, T, N))
    comb = torch.softmax(torch.randn(*lead, T, N, N), dim=-1)
    for _ in range(10):  # roughly doubly stochastic, like the hc step's output
        comb = comb / comb.sum(dim=-2, keepdim=True)
        comb = comb / comb.sum(dim=-1, keepdim=True)
    return f, x, post.float(), comb.reshape(*lead, T, N * N).float().contiguous()


def to_dev(t, device, dtype=ttnn.float32):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def run(device, lead, T, C, **kwargs):
    f, x, post, comb = make_inputs(lead, T, C)
    out = mhc_post(to_dev(f, device), to_dev(x, device), to_dev(post, device), to_dev(comb, device), **kwargs)
    return out, torch_mhc_post(f, x, post, comb)


SHAPES = [
    pytest.param((), 32, 32, id="single_tile_rank2"),
    pytest.param((), 64, 256, id="multi_tile_rank2"),
    pytest.param((1,), 128, 1024, id="non_square_rank3"),
    pytest.param((1, 2), 64, 256, id="multi_batch_rank4"),
    pytest.param((1, 1), 160, 5120, id="ds41_rows_rank4"),
    pytest.param((1,), 47, 128, id="t_not_tile_aligned"),
    pytest.param((), 1, 1792, id="single_token"),
]


@pytest.mark.parametrize("lead,T,C", SHAPES)
def test_mhc_post_plain_fp32(device, lead, T, C):
    out, ref = run(device, lead, T, C)
    assert out.dtype == ttnn.float32
    assert out.layout == ttnn.TILE_LAYOUT
    got = ttnn.to_torch(out)
    assert tuple(got.shape) == tuple(ref.shape)
    p = pcc(got, ref)
    assert p >= PCC[ttnn.float32], f"PCC {p}"
    # fp32 streams are mixed in fp32 (SFPU): far tighter than PCC; guard against a tf32 FPU path.
    rel = ((got.double() - ref).norm() / ref.norm()).item()
    assert rel < 2e-6, f"relative error {rel}"


def test_mhc_post_identity_passthrough(device):
    """post = 0, comb = I: X' must equal X bit-for-bit (pins comb orientation and exact broadcast)."""
    torch.manual_seed(42)
    T, C = 64, 256
    f = torch.randn(T, C)
    x = torch.randn(T, N * C)
    post = torch.zeros(T, N)
    comb = torch.eye(N).expand(T, N, N).reshape(T, N * N).contiguous()
    out = mhc_post(to_dev(f, device), to_dev(x, device), to_dev(post, device), to_dev(comb, device))
    assert torch.equal(ttnn.to_torch(out), x)


def test_mhc_post_cyclic_comb(device):
    """comb[i][(i+1) % n] = 1: X'_j carries X_{j-1} (transposed mixing)."""
    torch.manual_seed(42)
    T, C = 32, 128
    f = torch.randn(T, C)
    x = torch.randn(T, N * C)
    post = torch.ones(T, N)
    comb = torch.roll(torch.eye(N), shifts=1, dims=1).expand(T, N, N).reshape(T, N * N).contiguous()
    out = ttnn.to_torch(mhc_post(to_dev(f, device), to_dev(x, device), to_dev(post, device), to_dev(comb, device)))
    ref = torch_mhc_post(f, x, post, comb)
    assert pcc(out, ref) >= PCC[ttnn.float32]
    assert ((out.double() - ref).norm() / ref.norm()).item() < 2e-6


def test_mhc_post_deterministic_and_grid_invariant(device):
    """Two calls give the same bits; a 3 x 2 max_grid gives the same X' bits as the full grid."""
    f, x, post, comb = make_inputs((1, 1), 160, 1024)
    args = [to_dev(t, device) for t in (f, x, post, comb)]
    a = ttnn.to_torch(mhc_post(*args))
    b = ttnn.to_torch(mhc_post(*args))
    c = ttnn.to_torch(mhc_post(*args, max_grid=ttnn.CoreCoord(3, 2)))
    assert torch.equal(a, b)
    assert torch.equal(a, c)


def test_mhc_post_output_tensor(device):
    """The caller's buffer is written and returned."""
    f, x, post, comb = make_inputs((), 64, 256)
    args = [to_dev(t, device) for t in (f, x, post, comb)]
    dst = to_dev(torch.zeros_like(x), device)
    out = mhc_post(*args, output_tensor=dst)
    assert out.buffer_address() == dst.buffer_address()
    ref = torch_mhc_post(f, x, post, comb)
    assert pcc(ttnn.to_torch(dst), ref) >= PCC[ttnn.float32]

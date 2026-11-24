# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Device unit tests for the mHC (hyper-connection) block.

The reference is the formula in the module docstring, which is GLM's
``modeling_glm5_next.py:296-329`` and identical to DeepSeek-V4's mHC:

    m     = rsqrt(mean(x^2) + norm_eps) * (x @ fn^T)
    pre   = sigmoid(scale0 * m[0:4]  + base[0:4])  + hc_eps
    post  = 2 * sigmoid(scale1 * m[4:8] + base[4:8])
    C     = Sinkhorn_iters(softmax_row(scale2 * m[8:24].view(4,4) + base[8:24]) + hc_eps)
    h     = sum_i pre_i X_i
    X'_j  = post_j * y + sum_i C[i, j] X_i

The Sinkhorn half runs on the existing ``deepseek_prefill.mhc_split_sinkhorn`` kernel, which was
verified equal to GLM modeling:313-324 upstream; this test grades the whole block against a
torch Sinkhorn written from the same definition.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.experimental.glm53_flash.tt.hyperconnection import HyperConnection, build_sinkhorn_consts

pytestmark = pytest.mark.use_module_device({"l1_small_size": 24576})

DRAM = ttnn.DRAM_MEMORY_CONFIG
HC = 4
HIDDEN = 4096  # the block only supports the checkpoint width (fn is [24, hc*hidden])


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def sinkhorn_knopp(m: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    """Row-softmax then ``iters`` alternating row / column normalizations, on the last two dims."""
    n = m.shape[-1]
    logits = m + eps
    p = torch.softmax(logits, dim=-1)
    for _ in range(iters):
        p = p / (p.sum(-2, keepdim=True).clamp_min(1e-20))
        p = p / (p.sum(-1, keepdim=True).clamp_min(1e-20))
    return p


def mhc_reference(x, fn, base, scale, iters, hc_eps, norm_eps, y=None):
    """``x [T, hc*H]`` streams -> (h [T,H], post [T,hc], comb [T,hc,hc], new_streams [T,hc*H])."""
    T = x.shape[0]
    n = HC
    inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm_eps)  # unweighted RMS over hc*H
    m = (x @ fn.t()) * inv  # [T, (2+hc)*hc] = 24
    pre = torch.sigmoid(scale[0] * m[:, :n] + base[:n]) + hc_eps
    post = 2 * torch.sigmoid(scale[1] * m[:, n : 2 * n] + base[n : 2 * n])
    cm = scale[2] * m[:, 2 * n :].view(T, n, n) + base[2 * n :].view(n, n)
    comb = sinkhorn_knopp(cm, iters, hc_eps)

    X = x.view(T, n, HIDDEN)
    h = (pre.unsqueeze(-1) * X).sum(1)  # [T, H]
    if y is None:
        return h, post, comb, None
    # X'_j = post_j * y + sum_i C[i,j] X_i
    # einsum contracts residual-stream index i; comb is already [i, j] so no transpose.
    new = post.unsqueeze(-1) * y.unsqueeze(1) + torch.einsum("tij,tih->tjh", comb, X)
    return h, post, comb, new.reshape(T, n * HIDDEN)


def _make(device, iters=20, hc_eps=1e-6, norm_eps=1e-5, seed=0):
    torch.manual_seed(seed)
    fn = torch.randn((2 + HC) * HC, HC * HIDDEN) * 0.02  # [24, 16384], the checkpoint's layout
    base = torch.randn((2 + HC) * HC) * 0.1
    scale = torch.randn(3) * 0.1
    # The module takes fn^T [16384, 24] (see its __init__), i.e. the transposed checkpoint tensor.
    hc = HyperConnection(
        device,
        ttnn.from_torch(
            fn.t().contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM
        ),
        base,
        scale,
        hc_mult=HC,
        hidden_size=HIDDEN,
        sinkhorn_iters=iters,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
    )
    return hc, fn.float(), base, scale


def test_sinkhorn_consts_shape(device):
    """The packed constants table is the 8-row block the split op reads."""
    fn = torch.zeros((2 + HC) * HC, HC * HIDDEN)
    base = torch.randn((2 + HC) * HC)
    scale = torch.randn(3)
    consts = build_sinkhorn_consts(HC, scale, base)
    assert consts.shape[0] == 8
    assert consts.dtype == torch.float32


@pytest.mark.parametrize("tokens", [1, 32])
def test_hyperconnection_pre(device, tokens):
    """``pre`` reproduces the collapse and returns post / comb of the right shape."""
    hc, fn, base, scale = _make(device)
    x = torch.randn(tokens, HC * HIDDEN) * 0.5
    h, post, comb = hc.pre(
        ttnn.from_torch(
            x.reshape(1, 1, tokens, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM
        )
    )
    ttnn.synchronize_device(device)

    ref_h, ref_post, ref_comb, _ = mhc_reference(x, fn, base, scale, 20, 1e-6, 1e-5)

    got_h = ttnn.to_torch(h).float().reshape(tokens, HIDDEN)
    got_post = ttnn.to_torch(post).float().reshape(tokens, HC)
    got_comb = ttnn.to_torch(comb).float().reshape(tokens, HC, HC)

    assert pcc(got_h, ref_h) > 0.99, f"collapse PCC {pcc(got_h, ref_h):.5f}"
    # post / comb are gating quantities; bf16-ish tolerance on PCC rather than exact equality.
    assert pcc(got_post, ref_post) > 0.99, f"post PCC {pcc(got_post, ref_post):.5f}"
    assert pcc(got_comb, ref_comb) > 0.99, f"comb PCC {pcc(got_comb, ref_comb):.5f}"


def test_sinkhorn_is_doubly_stochastic(device):
    """A Sinkhorn-constrained ``comb`` must have rows and columns summing to 1 (that is the point)."""
    hc, fn, base, scale = _make(device, seed=3)
    x = torch.randn(32, HC * HIDDEN) * 0.5
    _, _, comb = hc.pre(
        ttnn.from_torch(
            x.reshape(1, 1, 32, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM
        )
    )
    ttnn.synchronize_device(device)
    c = ttnn.to_torch(comb).float().reshape(32, HC, HC)
    rows = c.sum(-1)
    cols = c.sum(-2)
    assert torch.allclose(rows, torch.ones_like(rows), atol=5e-2), rows
    assert torch.allclose(cols, torch.ones_like(cols), atol=5e-2), cols


@pytest.mark.parametrize("tokens", [1, 32])
def test_hyperconnection_post(device, tokens):
    """``post`` mixes the residual streams with the sublayer output: X'_j = post_j y + sum_i C[i,j] X_i."""
    hc, fn, base, scale = _make(device, seed=1)
    x = torch.randn(tokens, HC * HIDDEN) * 0.5
    y = torch.randn(tokens, HIDDEN) * 0.5

    xt = ttnn.from_torch(
        x.reshape(1, 1, tokens, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM
    )
    _, post, comb = hc.pre(xt)
    yt = ttnn.from_torch(
        y.reshape(1, 1, tokens, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM
    )
    out = hc.post(yt, xt, post, comb)
    ttnn.synchronize_device(device)

    _, _, _, ref_new = mhc_reference(x, fn, base, scale, 20, 1e-6, 1e-5, y=y)
    got = ttnn.to_torch(out).float().reshape(tokens, HC * HIDDEN)
    assert pcc(got, ref_new) > 0.99, f"post PCC {pcc(got, ref_new):.5f}"


def test_hyperconnection_pre_does_not_mutate_input(device):
    """The layer contracts pass streams and pre_mix by reference, so ``pre`` must leave x intact."""
    hc, fn, base, scale = _make(device, seed=2)
    x = torch.randn(8, HC * HIDDEN) * 0.5
    xt = ttnn.from_torch(
        x.reshape(1, 1, 8, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM
    )
    hc.pre(xt)
    ttnn.synchronize_device(device)
    assert torch.allclose(ttnn.to_torch(xt).float().reshape(8, HC * HIDDEN), x, rtol=0, atol=0)

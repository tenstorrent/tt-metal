# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision baseline for mhc_post (verifier, Phase 0: plain, fp32 streams + fp32 sublayer, separate).

Per shape: PCC (assert_with_pcc), comp_allclose, max / mean abs error, relative RMS error,
fp32 ULP distance (p50 / p99 / max) against the float64 reference rounded once to fp32,
the signed relative bias (the golden suite's depth gate), and the got/true ratio spread
(median, p5 / p95) — the scale-bug detector: a tight cluster around a non-1.0 constant
would be a structural scale bug, a tight cluster on 1.0 is rounding noise.
Run with `-s` to see the printed table rows.
"""
import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.mhc_post import mhc_post

N = 4


def _sinkhorn(logits, iters=20, eps=1e-6):
    m = torch.softmax(logits, dim=-1) + eps
    m = m / (m.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    return m


def _dev(t, device):
    return ttnn.from_torch(
        t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _ulp_distance(a32, b32):
    """|a - b| in fp32 ULPs (monotone int mapping of the bit patterns)."""

    def ordered(t):
        i = t.contiguous().view(torch.int32).to(torch.int64)
        return torch.where(i < 0, -(i & 0x7FFFFFFF), i)

    return (ordered(a32) - ordered(b32)).abs()


SHAPES = [
    (32, 4 * 32),  # small, one tile per stream
    (1, 128, 4 * 1024),  # medium, rank 3
    (1, 1, 150, 4 * 5120),  # DeepSeek-V4.1 rows, ragged T
    (1, 1, 640, 4 * 7168),  # large: DeepSeek-V4 per-device prefill, C = 7168
]


@pytest.mark.parametrize("x_shape", SHAPES, ids=["x".join(map(str, s)) for s in SHAPES])
def test_precision_baseline(x_shape, device):
    torch.manual_seed(1234)
    lead, nc = tuple(x_shape[:-1]), x_shape[-1]
    C = nc // N
    f = torch.randn(*lead, C)
    x = torch.randn(*lead, nc)
    post = 2.0 * torch.sigmoid(torch.randn(*lead, N))
    comb = _sinkhorn(torch.randn(*lead, N, N)).reshape(*lead, N * N).contiguous()

    out = mhc_post(_dev(f, device), _dev(x, device), _dev(post, device), _dev(comb, device))
    got = ttnn.to_torch(out).float()

    ff = f.double().reshape(-1, 1, C)
    xx = x.double().reshape(-1, N, C)
    pp = post.double().reshape(-1, N, 1)
    mm = comb.double().reshape(-1, N, N)
    ref64 = (pp * ff + torch.einsum("tij,tic->tjc", mm, xx)).reshape(*lead, nc)
    ref32 = ref64.float()

    assert_with_pcc(ref32, got, 0.99999)
    _, allclose_msg = comp_allclose(ref32, got)

    diff = got.double() - ref64
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    rel_rms = diff.pow(2).mean().sqrt().item() / ref64.std().item()
    ulp = _ulp_distance(got, ref32).double().flatten()
    ulp_p50, ulp_p99 = torch.quantile(ulp[: 1 << 24], torch.tensor([0.5, 0.99], dtype=torch.float64)).tolist()
    d = (diff * ref64.sign()).flatten()
    bias = (d.mean() / ref64.abs().mean()).item()

    nz = ref64.abs() > 1e-3
    ratio = (got.double()[nz] / ref64[nz]).flatten()
    sample = ratio[torch.randperm(ratio.numel())[: 1 << 22]]
    r_p5, r_med, r_p95 = torch.quantile(sample, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)).tolist()

    print(
        f"\nBASELINE {x_shape}: {allclose_msg} | max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} "
        f"rel_rms={rel_rms:.3e} ulp_p50={ulp_p50:.0f} ulp_p99={ulp_p99:.0f} ulp_max={ulp.max().item():.0f} "
        f"bias={bias:+.2e} ratio_med={r_med:.9f} ratio_p5={r_p5:.9f} ratio_p95={r_p95:.9f}"
    )
    # Phase 0 contract: fp32 X' exact to ~2e-6 relative RMS and unbiased (golden gates).
    assert rel_rms <= 2e-6
    assert abs(bias) <= 1e-6
    assert abs(r_med - 1.0) < 1e-6

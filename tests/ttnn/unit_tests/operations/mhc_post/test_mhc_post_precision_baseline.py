# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision baseline for mhc_post (Phase 0 SUPPORTED: plain mode, fp32 streams + sublayer, separate coefs).

Measures PCC, max / mean abs error, relative RMS, the fp32 ULP error against a float64 reference, and the
got/true ratio spread (scale-bug detector: a tight cluster at a non-1.0 constant is a structural bug, a broad
cluster centred on 1.0 is rounding noise). Run with `-s` to see the table.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, ulp
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.mhc_post import mhc_post

N = 4


def reference(f, x, post, comb):
    lead, C = tuple(f.shape[:-1]), f.shape[-1]
    f3 = f.double().reshape(-1, 1, C)
    x3 = x.double().reshape(-1, N, C)
    p3 = post.double().reshape(-1, N, 1)
    m = comb.double().reshape(-1, N, N)
    return (p3 * f3 + torch.einsum("tij,tic->tjc", m, x3)).reshape(*lead, N * C)


def make_inputs(lead, T, C, seed=7):
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(*lead, T, C, generator=g)
    x = torch.randn(*lead, T, N * C, generator=g)
    post = 2.0 * torch.sigmoid(torch.randn(*lead, T, N, generator=g))
    comb = torch.softmax(torch.randn(*lead, T, N, N, generator=g), dim=-1)
    for _ in range(20):  # doubly stochastic, like the hc step's output
        comb = comb / comb.sum(dim=-2, keepdim=True)
        comb = comb / comb.sum(dim=-1, keepdim=True)
    return f, x, post.float(), comb.reshape(*lead, T, N * N).float().contiguous()


SHAPES = [
    pytest.param((), 32, 128, id="small_T32_C128"),
    pytest.param((1,), 100, 1024, id="medium_T100_C1024_nonaligned"),
    pytest.param((1, 1), 160, 5120, id="ds41_T160_C5120"),
    pytest.param((1, 1), 640, 1792, id="large_T640_C1792"),
]


@pytest.mark.parametrize("lead,T,C", SHAPES)
def test_mhc_post_precision_baseline(device, lead, T, C):
    f, x, post, comb = make_inputs(lead, T, C)
    dev = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(mhc_post(dev(f), dev(x), dev(post), dev(comb))).double()
    ref = reference(f, x, post, comb)

    _, pcc_msg = assert_with_pcc(ref, got, pcc=0.99999)
    _, allclose_msg = comp_allclose(ref, got)
    err = (got - ref).abs()
    rel_rms = ((got - ref).norm() / ref.norm()).item()
    ref32 = ref.float()
    ulp_err = (got.float() - ref32).abs() / ulp(ref32)
    nz = ref.abs() > 1e-6
    r = got[nz] / ref[nz]
    q = torch.quantile(r.float()[: 1 << 24], torch.tensor([0.05, 0.5, 0.95])).tolist()
    print(
        f"\n[precision] lead={lead} T={T} C={C}: {pcc_msg} | max_abs={err.max().item():.3e} "
        f"mean_abs={err.mean().item():.3e} rel_rms={rel_rms:.3e} | ulp p50={ulp_err.median().item():.2f} "
        f"p99={torch.quantile(ulp_err.flatten()[: 1 << 24], 0.99).item():.2f} max={ulp_err.max().item():.1f} | "
        f"ratio p5={q[0]:.9f} median={q[1]:.9f} p95={q[2]:.9f} | {allclose_msg}"
    )
    # fp32 SFPU mixing: one fp32 rounding per MAD, against a float64 reference
    assert rel_rms < 2e-6, rel_rms
    assert abs(q[1] - 1.0) < 1e-6, f"ratio median {q[1]} (scale bug?)"

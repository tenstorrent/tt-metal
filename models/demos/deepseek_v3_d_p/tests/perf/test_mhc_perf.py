# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-perf for mHC: (1) parametrization naive vs fused kernel, (2) the hc_post computation.

(1) test_mhc_parametrize_naive_vs_fused — same input `mixes` [T, (2+n)*n], same output
(pre, post, comb). Naive = TtMHC.parametrize (the split + affine + sigmoid + matmul-selection
Sinkhorn as ~85 separate ttnn ops, each round-tripping DRAM); fused = the single
mhc_split_sinkhorn kernel. Signposts "mhc-naive" / "mhc-fused".

(2) test_mhc_hc_post_perf — profiles the *computation* half (hc_post) at realistic C. Measured
breakdown: ~81% ttnn matmul (two [1,T,n,C] batched matmuls, both at ~55% of peak DRAM BW),
~19% the add, <0.5% reshapes. It is bandwidth-bound, so it is left as a composite on purpose:
the only fusible win is the add, giving a ~1.2x ceiling on a custom kernel that would otherwise
just reimplement — and likely lose to — ttnn's already-tuned matmul. Signpost "mhc-hc-post".

Profile with:  python -m tracy -r -p -v -m pytest <this file> -k <id>
then tt-perf-report on the CSV; the signposted regions delimit the op sets to sum device time over.
"""

import pytest
import torch
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import MHCConfig
from models.demos.deepseek_v3_d_p.tt.mhc.mhc_kernel import build_consts
from models.demos.deepseek_v3_d_p.tt.mhc.mhc_ttnn import TtMHC

ITERS = 3  # measured dispatches per variant (after a warmup); keep small so the device
# profiler's marker buffer doesn't overflow on the ~85-op naive path


@pytest.mark.parametrize("T", [2048, 8192], ids=["T2048", "T8192"])
def test_mhc_parametrize_naive_vs_fused(device, T):
    torch.manual_seed(0)
    cfg = MHCConfig(dim=7168, n=4)  # V4-Pro C; the kernel is hidden-dim-independent (acts on
    # mixes[T,24]), so dim only sizes the projection weight that this test never exercises
    g = torch.Generator().manual_seed(1)
    mixes = torch.randn(T, cfg.mix_hc, generator=g)
    scale = torch.full((3,), 1.0)
    base = torch.randn(cfg.mix_hc, generator=g)
    fn = torch.randn(cfg.mix_hc, cfg.n * cfg.dim, generator=g) * 0.02

    mhc = TtMHC(device, cfg, fn, base, scale)  # naive parametrize + uploaded RB/CB/EMB/biases
    consts = ttnn.from_torch(build_consts(cfg, scale, base), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    mixes_naive = ttnn.from_torch(
        mixes.reshape(1, 1, T, cfg.mix_hc), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32
    )
    mixes_fused = ttnn.from_torch(mixes, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)

    def naive():
        return mhc.parametrize(mixes_naive)

    def fused():
        return ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
            mixes_fused, consts, cfg.n, int(cfg.sinkhorn_iters), float(cfg.eps)
        )

    # warmup: compile kernels + populate program cache for both paths
    naive()
    fused()
    ttnn.synchronize_device(device)

    signpost("mhc-naive")
    for _ in range(ITERS):
        naive()
    ttnn.synchronize_device(device)

    signpost("mhc-fused")
    for _ in range(ITERS):
        fused()
    ttnn.synchronize_device(device)


@pytest.mark.parametrize("T", [2048, 8192], ids=["T2048", "T8192"])
def test_mhc_hc_post_perf(device, T):
    """Device-perf of the *computation* half (hc_post), at realistic C, to see where its
    time goes: two [1,T,n,C] batched matmuls (term1 = post outer F, term2 = comb^T @ X) plus
    the [1,T,n,C] add. Unlike parametrize this is big-data / bandwidth-bound, so the numbers
    tell us whether a fused hc_post kernel could beat ttnn's tuned matmul + a cheap add."""
    torch.manual_seed(0)
    cfg = MHCConfig(dim=7168, n=4)
    C, n = cfg.dim, cfg.n
    g = torch.Generator().manual_seed(1)
    scale = torch.full((3,), 1.0)
    base = torch.randn(cfg.mix_hc, generator=g)
    fn = torch.randn(cfg.mix_hc, n * C, generator=g) * 0.02  # only feeds project(), unused here
    mhc = TtMHC(device, cfg, fn, base, scale)

    def up(t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)

    f_out = up(torch.randn(1, T, 1, C, generator=g))
    x_streams = up(torch.randn(1, T, n, C, generator=g))
    post = up(2.0 * torch.sigmoid(torch.randn(1, 1, T, n, generator=g)))
    comb = up(torch.rand(1, T, n, n, generator=g))  # perf only; need not be doubly-stochastic

    def hc_post():
        return mhc.hc_post(f_out, x_streams, post, comb)

    hc_post()  # warmup: compile + program cache
    ttnn.synchronize_device(device)

    signpost("mhc-hc-post")
    for _ in range(ITERS):
        hc_post()
    ttnn.synchronize_device(device)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-perf: naive (stock ttnn ops) vs fused-kernel mHC parametrization.

Same input `mixes` [T, (2+n)*n], same output (pre, post, comb). The naive path is
TtMHC.parametrize — the split + affine + sigmoid + matmul-selection Sinkhorn as ~85 separate
ttnn ops (each round-tripping DRAM). The fused path is the single mhc_split_sinkhorn kernel.

Profile with:  python -m tracy -r -p -v -m pytest <this file> -k <iters/T id>
then tt-perf-report on the CSV; the two signposted regions ("mhc-naive" / "mhc-fused")
delimit the op sets to sum device time over.
"""

import pytest
import torch
from tracy import signpost

import ttnn
from models.demos.deepseek_v4.reference.mhc_reference import MHCConfig
from models.demos.deepseek_v4.tt.mhc_kernel import build_consts
from models.demos.deepseek_v4.tt.mhc_ttnn import TtMHC

ITERS = 3  # measured dispatches per variant (after a warmup); keep small so the device
# profiler's marker buffer doesn't overflow on the ~85-op naive path


@pytest.mark.parametrize("T", [2048, 8192], ids=["T2048", "T8192"])
def test_mhc_parametrize_naive_vs_fused(device, T):
    torch.manual_seed(0)
    cfg = MHCConfig(dim=64, n=4)  # dim only feeds the (unused-here) projection weight
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

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-perf for mHC. Local measurement only -- these assert nothing and are in no pipeline.

Run under tracy and sum device time per signposted region:

    python -m tracy -r -p -v -m pytest models/demos/deepseek_v3_d_p/tests/perf/test_mhc_perf.py -k <id>

Token counts are per device. V4 prefills in 5k chunks, sequence-parallel over the 8-wide axis of
an 8x4 Galaxy mesh, so the shape that ships is T=5120/8 = 640 -- 20 token-tiles. Because a token's
H matrices depend only on that token's own mixes, the chunk size *is* the token count regardless of
total context: a 1M prompt runs the same 640 as a 5k one. T5120 is the chunk unsharded, T8192
reproduces an earlier measurement, and T131072 only probes what the kernel does when not starved.

One test per question:
  - kernel_grid_scaling: cost of the fused parametrization at prefill token counts, and whether
    it scales with the core grid (MHC_MAX_CORES=1 pins the single-core arm).
  - kernel_sharded: whether the zero-copy L1-sharded path beats DRAM-interleaved at equal work.
  - block: how the kernel compares to the composite ttnn half it sits in, at V4-Pro C=7168.
"""

import os

import pytest
import torch
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import MHCConfig
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import TtMHCWrap, build_consts

ITERS = 10  # measured dispatches per region, after a warmup that compiles and fills the cache


def _consts(device, cfg, scale, base):
    return ttnn.from_torch(build_consts(cfg, scale, base), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)


def _params(cfg, T, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(T, cfg.mix_hc, generator=g),
        torch.full((3,), 1.0),
        torch.randn(cfg.mix_hc, generator=g),
    )


# The kernel is hidden-dim independent -- it reads mixes[T, (2+n)*n] -- so dim only sizes the
# projection weight that these two kernel-only tests never build.
_KERNEL_CFG = MHCConfig(dim=64, n=4)


@pytest.mark.parametrize("T", [640, 5120, 8192, 131072], ids=["T640", "T5120", "T8192", "T131072"])
def test_mhc_kernel_grid_scaling(device, T):
    """Fused parametrization alone, full grid vs one core, at equal work.

    Token-tiles (T/32) are split across the compute grid, so the single-core arm is the
    serial cost of the same work and the ratio is the achieved parallel efficiency. At the
    shipping T=640 there are only 20 tiles to spread over ~130 cores, so that ratio is capped
    by the tile count, not by the grid.
    """
    torch.manual_seed(0)
    cfg = _KERNEL_CFG
    mixes, scale, base = _params(cfg, T)
    consts = _consts(device, cfg, scale, base)
    mixes_tt = ttnn.from_torch(mixes, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)

    def run():
        return ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
            mixes_tt, consts, cfg.n, int(cfg.sinkhorn_iters), float(cfg.eps)
        )

    run()  # compile + program cache
    ttnn.synchronize_device(device)
    signpost("mhc-kernel-grid")
    for _ in range(ITERS):
        run()
    ttnn.synchronize_device(device)

    if T > 8192:
        return  # the serial arm scales linearly; two token counts are enough to fix the ratio

    # max_cores rides in the hashed attributes, so flipping it re-keys the program cache
    # rather than reusing the multi-core program.
    prev = os.environ.get("MHC_MAX_CORES")
    os.environ["MHC_MAX_CORES"] = "1"
    try:
        run()
        ttnn.synchronize_device(device)
        signpost("mhc-kernel-1core")
        for _ in range(ITERS):
            run()
        ttnn.synchronize_device(device)
    finally:
        if prev is None:
            os.environ.pop("MHC_MAX_CORES", None)
        else:
            os.environ["MHC_MAX_CORES"] = prev


@pytest.mark.parametrize("tiles_per_core", [4], ids=["tpc4"])
def test_mhc_kernel_sharded(device, tiles_per_core):
    """DRAM-interleaved vs L1-height-sharded input at identical token count and grid.

    The sharded path aliases the input and output CBs straight to the shards, so it pays no
    DRAM round-trip; this is the only measurement that shows what that is worth.
    """
    torch.manual_seed(0)
    cfg = _KERNEL_CFG
    grid = device.compute_with_storage_grid_size()
    cores = grid.x * grid.y
    T = cores * tiles_per_core * 32
    mixes, scale, base = _params(cfg, T)
    consts = _consts(device, cfg, scale, base)

    inter = ttnn.from_torch(mixes, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    # sharded TILE tensors need a tile-aligned shard width, so mixes is padded 24 -> 32
    mixes32 = torch.zeros(T, 32)
    mixes32[:, : cfg.mix_hc] = mixes
    mem = ttnn.create_sharded_memory_config(
        [T, 32],
        ttnn.CoreGrid(y=grid.y, x=grid.x),
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    shard = ttnn.from_torch(mixes32, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32, memory_config=mem)

    def run(t):
        return ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
            t, consts, cfg.n, int(cfg.sinkhorn_iters), float(cfg.eps)
        )

    for arm, header in ((inter, "mhc-kernel-interleaved"), (shard, "mhc-kernel-sharded")):
        run(arm)
        ttnn.synchronize_device(device)
        signpost(header)
        for _ in range(ITERS):
            run(arm)
        ttnn.synchronize_device(device)


# n=4 is the tile-row dim of [1,T,n,C], so every stage tensor is padded to n=32 and costs 8x its
# logical size -- 587 MB per tensor at T=640/C=7168/fp32, and DRAM is exhausted by T=4096.
@pytest.mark.parametrize("T", [640, 2048], ids=["T640", "T2048"])
def test_mhc_block_perf(device, T):
    """The four stages of an mHC-wrapped sublayer at V4-Pro C, each in its own region.

    project and hc_post move [1,T,n,C] data and are bandwidth-bound; the fused kernel and the
    hc_pre reduction touch only [T,32] and [1,T,n,C] respectively. Splitting them is what shows
    whether the parametrization is worth optimising further or already lost in the noise.
    """
    torch.manual_seed(0)
    cfg = MHCConfig(dim=7168, n=4)
    C, n = cfg.dim, cfg.n
    g = torch.Generator().manual_seed(1)
    scale = torch.full((3,), 1.0)
    base = torch.randn(cfg.mix_hc, generator=g)
    fn = torch.randn(cfg.mix_hc, n * C, generator=g) * 0.02
    wrap = TtMHCWrap(device, cfg, fn, base, scale)

    def up(t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)

    x = up(torch.randn(1, T, n, C, generator=g))
    f_out = up(torch.randn(1, T, 1, C, generator=g))
    post = up(2.0 * torch.sigmoid(torch.randn(1, 1, T, n, generator=g)))
    comb = up(torch.rand(1, T, n, n, generator=g))  # perf only; need not be doubly-stochastic

    mixes = wrap.project(x)

    stages = (
        ("mhc-project", lambda: wrap.project(x)),
        (
            "mhc-kernel",
            lambda: ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
                mixes, wrap.consts, wrap.n, wrap.iters, wrap.eps
            ),
        ),
        ("mhc-hc-pre", lambda: wrap.hc_pre(x)),
        ("mhc-hc-post", lambda: wrap.hc_post(f_out, x, post, comb)),
    )
    for header, fn_ in stages:
        fn_()  # compile + program cache
        ttnn.synchronize_device(device)
        signpost(header)
        for _ in range(ITERS):
            fn_()
        ttnn.synchronize_device(device)

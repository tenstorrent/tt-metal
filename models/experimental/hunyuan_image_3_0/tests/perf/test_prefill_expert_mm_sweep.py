# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Config sweep for the two PREFILL MoE expert matmuls that dominate device time.

Profile (perfarpre1.txt, Blackhole 2x2, S=512 padded prefill) — matmul device time:
    512 x 4096 x 6144 (gate_up)  12811 us  (64% of all matmul time)  82% FLOP
    512 x 3072 x 4096 (down)      6340 us  (31%)                     78% FLOP
    everything else               ~1000 us (5%)  -> not worth tuning

Both run through wide_mm_program_config at M=512 (Mt=16 >= 8). On the 11x10=110
core Blackhole grid, wide_mm picks gy=largest_div(Mt<=10)=8, gx=largest_div(Nt<=11)=8
=> 64 cores. 46 of 110 cores sit idle because the config demands EXACT divisors of
Mt and Nt. This sweep asks: does spreading these onto more cores (ceil distribution,
1D split-N) or different subblocking beat the 64-core wide_mm baseline?

    pytest models/experimental/hunyuan_image_3_0/tests/perf/test_prefill_expert_mm_sweep.py -s
"""
from __future__ import annotations

import math
import time

import pytest
import torch
import ttnn
from models.experimental.hunyuan_image_3_0.tt.parallel_utils import wide_mm_program_config

_TILE = 32

# (name, K, N) for the two dominant prefill expert matmuls, plus the three smaller
# "SLOW"-flagged prefill projections (attn qkv, o_proj, MoE router gate).
SHAPES = [
    ("gate_up", 4096, 6144),
    ("down", 3072, 4096),
    ("attn_qkv", 4096, 3072),
    ("o_proj", 2048, 4096),
    ("gate_router", 4096, 64),
]

# Tracy per-shape profiler hint: (K, N) -> (in0_block_w, out_subblock_h, out_subblock_w).
_TRACY_HINT = {
    (4096, 3072): (4, 1, 3),
    (2048, 4096): (4, 1, 4),
    (4096, 64): (4, 2, 1),
}


def _bench(dev, fn, it=30):
    for _ in range(5):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t = time.perf_counter()
    for _ in range(it):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t) / it * 1e6


def _pcc(a, b):
    a = a.float().flatten() - a.float().mean()
    b = b.float().flatten() - b.float().mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


def _div_leq(x, cap):
    return next((d for d in range(min(x, cap), 0, -1) if x % d == 0), 1)


def _subblocks(pcm, pcn):
    osw = next((w for w in (4, 3, 2, 1) if pcn % w == 0), 1)
    osh = next((h for h in range(4 // osw, 0, -1) if pcm % h == 0), 1)
    return osh, osw


def _cfg_2d(gx, gy, Mt, Kt, Nt, ibw, *, exact):
    """2D-mcast on gx*gy cores. exact=True requires Mt%gy==Nt%gx==0 (no idle/ragged);
    exact=False uses ceil distribution so all gx*gy cores participate (last cores pad)."""
    if exact:
        if Mt % gy or Nt % gx:
            return None
        pcm, pcn = Mt // gy, Nt // gx
    else:
        pcm, pcn = math.ceil(Mt / gy), math.ceil(Nt / gx)
    osh, osw = _subblocks(pcm, pcn)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=ibw,
        out_subblock_h=osh,
        out_subblock_w=osw,
        per_core_M=pcm,
        per_core_N=pcn,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def _cfg_1d_split_n(gx, gy, Mt, Kt, Nt, ibw):
    """1D mcast, split N across all gx*gy cores (in0 broadcast to every core)."""
    ncores = gx * gy
    pcn = math.ceil(Nt / ncores)
    osw = next((w for w in (4, 3, 2, 1) if pcn % w == 0), 1)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=ibw,
        out_subblock_h=1,
        out_subblock_w=osw,
        per_core_M=Mt,
        per_core_N=pcn,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


@pytest.mark.parametrize("S", [512], ids=["seq512"])
@pytest.mark.parametrize("name,K,N", SHAPES, ids=[s[0] for s in SHAPES])
def test_prefill_expert_mm_sweep(device, S, name, K, N):
    device.enable_program_cache()
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    M = ((S + _TILE - 1) // _TILE) * _TILE
    torch.manual_seed(0)
    x = torch.randn(1, S, K, dtype=torch.bfloat16) * 0.05
    w = torch.randn(K, N, dtype=torch.bfloat16) * 0.02  # already transposed for ttnn.linear
    ref = x.float().reshape(-1, K) @ w.float()

    xt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    wt = ttnn.from_torch(
        w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    grid = device.compute_with_storage_grid_size()
    Mt, Kt, Nt = M // _TILE, K // _TILE, N // _TILE

    def mm(pc):
        return ttnn.linear(
            xt,
            wt,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=ckc,
            program_config=pc,
        )

    candidates = [
        ("wide_mm (PROD)", wide_mm_program_config(device, M, K, N)),
        ("auto", None),
    ]

    # Tracy's explicit profiler hint (in0_block_w + output subblock) on a few grids.
    if (K, N) in _TRACY_HINT:
        ibw_h, osh_h, osw_h = _TRACY_HINT[(K, N)]
        for gx in sorted({grid.x, _div_leq(Nt, grid.x), 8} & set(range(1, grid.x + 1))):
            for gy in sorted({_div_leq(Mt, grid.y), grid.y} & set(range(1, grid.y + 1))):
                pcm, pcn = math.ceil(Mt / gy), math.ceil(Nt / gx)
                if pcn % osw_h or pcm % osh_h:
                    continue
                hint = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(gx, gy),
                    in0_block_w=ibw_h,
                    out_subblock_h=osh_h,
                    out_subblock_w=osw_h,
                    per_core_M=pcm,
                    per_core_N=pcn,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=False,
                )
                candidates.append((f"TRACY {gx}x{gy}={gx*gy}c ibw={ibw_h} osb={osh_h}x{osw_h} pcn={pcn}", hint))

    # 2D exact-divisor grids: vary gy (rows over Mt) and gx (cols over Nt) + K chunk.
    for gy in sorted({_div_leq(Mt, grid.y), _div_leq(Mt, grid.y - 2), Mt, 1} & set(range(1, grid.y + 1))):
        for gx in sorted(
            {_div_leq(Nt, grid.x), _div_leq(Nt, grid.x - 1), _div_leq(Nt, grid.x // 2)} & set(range(1, grid.x + 1))
        ):
            for ibw in sorted({_div_leq(Kt, 2), _div_leq(Kt, 4), _div_leq(Kt, 8)}):
                c = _cfg_2d(gx, gy, Mt, Kt, Nt, ibw, exact=True)
                if c:
                    candidates.append(
                        (f"2D-exact {gx}x{gy}={gx*gy}c ibw={ibw} pcm={c.per_core_M} pcn={c.per_core_N}", c)
                    )

    # 2D ceil-distribution grids: use MORE cores (up to full 11x10) even when Mt/Nt
    # don't divide evenly. Try the widest grids.
    for gy in sorted({grid.y, grid.y - 2, Mt if Mt <= grid.y else grid.y}):
        for gx in sorted({grid.x, grid.x - 1, grid.x - 3}):
            if not (1 <= gy <= grid.y and 1 <= gx <= grid.x):
                continue
            for ibw in sorted({_div_leq(Kt, 2), _div_leq(Kt, 4)}):
                c = _cfg_2d(gx, gy, Mt, Kt, Nt, ibw, exact=False)
                if c:
                    candidates.append(
                        (f"2D-ceil {gx}x{gy}={gx*gy}c ibw={ibw} pcm={c.per_core_M} pcn={c.per_core_N}", c)
                    )

    # 1D split-N over wide grids (N is the big dim; broadcast the 16 M-tiles).
    for gx, gy in [(grid.x, grid.y), (grid.x, grid.y - 2), (8, 8)]:
        for ibw in sorted({_div_leq(Kt, 4), _div_leq(Kt, 8)}):
            candidates.append((f"1D-splitN {gx}x{gy}={gx*gy}c ibw={ibw}", _cfg_1d_split_n(gx, gy, Mt, Kt, Nt, ibw)))

    seen, base = set(), None
    results = []
    print(f"\n=== {name}: {M} x {K} x {N}  (Mt={Mt} Kt={Kt} Nt={Nt}, grid {grid.x}x{grid.y}) ===")
    for cname, pc in candidates:
        if pc is None:
            key = "auto"
        else:
            g = pc.compute_with_storage_grid_size
            key = ((g.x, g.y), pc.in0_block_w, pc.per_core_M, pc.per_core_N, type(pc).__name__)
        if key in seen:
            continue
        seen.add(key)
        try:
            out = ttnn.to_torch(mm(pc)).reshape(-1, N)
            us = _bench(device, lambda pc=pc: mm(pc))
            if base is None:
                base = us
            pcc = _pcc(out, ref)
            flag = "  <-- BAD PCC" if pcc < 0.99 else ""
            results.append((cname, us, pcc))
            print(f"  {cname:44} {us:8.1f} us  ({base/us:4.2f}x)  PCC={pcc:.5f}{flag}")
        except Exception as e:
            print(f"  {cname:44}    FAIL  {str(e)[:60]}")

    good = [r for r in results if r[2] >= 0.99]
    good.sort(key=lambda r: r[1])
    print(f"  --- BEST for {name}: {good[0][0]}  {good[0][1]:.1f}us ({base/good[0][1]:.2f}x vs wide_mm) ---")

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Config sweep for the MoE expert down projection matmul: M x 3072 x 4096
(HiFi2, BF16 act x BFP8 weight => BF16). This is the per-expert down_proj in
moe_parallel._expert, which ALREADY passes wide_mm_program_config. The sweep
checks whether that production config is actually the fastest for this shape,
A/B'd against auto and a grid of alternative 2D / 1D schedules, with PCC.

    pytest models/experimental/hunyuan_image_3_0/tests/perf/test_expert_down_sweep.py -s

Result (WH, program cache on, measured L1 & DRAM input): the winning schedule is
M-DEPENDENT, and wide_mm was being applied where it HURTS:
  * M=32 (Mt=1): 1D split-N ~1.55x vs auto.
  * M=64..224 (Mt 2-7): AUTO is 25-33%% faster than wide_mm. <- the profiled M=64
    shape was being pessimized 82.5us -> should be 62.5us on auto.
  * M>=256 (Mt>=8): wide_mm wins, and auto degrades catastrophically with M
    (L1 input: 5305us at M=2048 vs wide_mm 360us -- the documented mis-schedule).
So moe_parallel._expert now gates: wide_mm only for Mt>=8, else auto.
"""
from __future__ import annotations

import time

import pytest
import torch
import ttnn
from models.experimental.hunyuan_image_3_0.tt.parallel_utils import wide_mm_program_config

K, N = 3072, 4096  # intermediate, hidden
_TILE = 32


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


def _cfg_2d(grid, Mt, Kt, Nt, gy, gx, ibw):
    if Mt % gy or Nt % gx:
        return None
    pcm, pcn = Mt // gy, Nt // gx
    osw = next((w for w in (4, 3, 2, 1) if pcn % w == 0), 1)
    osh = next((h for h in range(4 // osw, 0, -1) if pcm % h == 0), 1)
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


def _cfg_1d_split_n(grid, Mt, Kt, Nt):
    ncores = grid.x * grid.y
    ncols = next(c for c in range(min(ncores, Nt), 0, -1) if Nt % c == 0)
    pcn = Nt // ncols
    osw = next(w for w in (4, 3, 2, 1) if pcn % w == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=_div_leq(Kt, 8),
        out_subblock_h=1,
        out_subblock_w=osw,
        per_core_M=Mt,
        per_core_N=pcn,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


@pytest.mark.parametrize("in_mem", ["L1", "DRAM"])
def test_expert_down_threshold(device, in_mem):
    """Map auto vs wide_mm vs 1D across the full M range and BOTH input mem states,
    to find the M threshold where wide_mm overtakes auto (and where auto's
    documented 1.3ms mis-schedule actually fires). This is what sets the gate in
    moe_parallel._expert."""
    device.enable_program_cache()
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    imc = ttnn.L1_MEMORY_CONFIG if in_mem == "L1" else ttnn.DRAM_MEMORY_CONFIG
    grid = device.compute_with_storage_grid_size()
    Kt, Nt = K // _TILE, N // _TILE
    torch.manual_seed(0)
    w = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
    wt = ttnn.from_torch(
        w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    print(f"\n=== down-proj M-threshold, input={in_mem} (K={K} N={N}) ===")
    print(f"  {'M':>5} {'auto':>10} {'wide_mm':>12} {'1D':>10}   winner")
    for M in (32, 64, 96, 128, 256, 512, 1024, 2048):
        Mt = M // _TILE
        x = torch.randn(1, M, K, dtype=torch.bfloat16) * 0.05
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=imc)

        def mm(pc):
            return ttnn.linear(
                xt, wt, dtype=ttnn.bfloat16, memory_config=imc, compute_kernel_config=ckc, program_config=pc
            )

        row = {}
        cfgs = {"auto": None, "wide_mm": wide_mm_program_config(device, M, K, N)}
        if Mt == 1:
            cfgs["1D"] = _cfg_1d_split_n(grid, Mt, Kt, Nt)
        for name, pc in cfgs.items():
            try:
                row[name] = _bench(device, lambda pc=pc: mm(pc))
            except Exception:
                row[name] = None
        best = min((v, k) for k, v in row.items() if v is not None)[1]
        fmt = lambda v: f"{v:9.1f}u" if v is not None else "     n/a "
        print(f"  {M:>5} {fmt(row.get('auto')):>10} {fmt(row.get('wide_mm')):>12} {fmt(row.get('1D')):>10}   <- {best}")


@pytest.mark.parametrize("S", [32, 64, 256], ids=["seq32_mt1", "seq64", "seq256"])
def test_expert_down_sweep(device, S):
    device.enable_program_cache()
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    M = ((S + _TILE - 1) // _TILE) * _TILE
    torch.manual_seed(0)
    x = torch.randn(1, S, K, dtype=torch.bfloat16) * 0.05
    w = torch.randn(K, N, dtype=torch.bfloat16) * 0.02  # [I, H]
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

    candidates = [("auto", None), ("wide_mm (prod)", wide_mm_program_config(device, M, K, N))]
    if Mt == 1:
        candidates.append(("1D split-N", _cfg_1d_split_n(grid, Mt, Kt, Nt)))
    for gy in (Mt, max(1, Mt // 2), 1):
        for gx in (_div_leq(Nt, grid.x), _div_leq(Nt, max(1, grid.x // 2))):
            for ibw in (_div_leq(Kt, 4), _div_leq(Kt, 8)):
                if gy <= grid.y and gx <= grid.x:
                    c = _cfg_2d(grid, Mt, Kt, Nt, gy, gx, ibw)
                    if c:
                        candidates.append((f"2D gy={gy} gx={gx} ibw={ibw}", c))

    seen, base = set(), None
    print(f"\n=== expert down {M} x {K} x {N} (S={S}) ===")
    for name, pc in candidates:
        key = (
            name
            if pc is None
            else (
                str(pc.compute_with_storage_grid_size),
                pc.in0_block_w,
                pc.per_core_M,
                pc.per_core_N,
                type(pc).__name__,
            )
        )
        if key in seen:
            continue
        seen.add(key)
        try:
            out = ttnn.to_torch(mm(pc)).reshape(-1, N)
            us = _bench(device, lambda pc=pc: mm(pc))
            if base is None:
                base = us
            print(f"  {name:26} {us:8.1f} us  ({base/us:4.2f}x)  PCC={_pcc(out, ref):.6f}")
        except Exception as e:
            print(f"  {name:26}    FAIL  {str(e)[:70]}")

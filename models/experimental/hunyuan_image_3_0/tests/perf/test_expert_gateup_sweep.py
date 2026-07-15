# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Config sweep for the MoE expert gate_up projection matmul: M x 4096 x 6144
(HiFi2, BF16 act x BFP8 weight => BF16). This is the per-expert gate_and_up
linear in moe_parallel._expert (the only expert matmul with no program_config;
the sibling down-proj already uses wide_mm_program_config).

Unlike the tiny router, this one is genuinely wide (N=6144) — auto already puts
it on ~96 cores at ~51%% DRAM / 33.6 TFLOPs, so the "SLOW" flag mostly means "no
explicit program_config". The sweep checks whether an explicit 2D-mcast grid (the
schedule the fast expert matmuls use) beats auto here.

    pytest models/experimental/hunyuan_image_3_0/tests/perf/test_expert_gateup_sweep.py -s

Result (WH, program cache on): AUTO IS FASTEST. Every explicit config is slower —
wide_mm_program_config 0.64x (S=64) / 0.92x (S=256), best 2D grid 0.96x. The SLOW
flag is a false alarm for this wide-N shape: auto's heuristic already picks a good
~96-core schedule, and forcing a program_config only hurts. So gate_up is
correctly left on auto in moe_parallel._expert — do NOT add a program_config here.
"""
from __future__ import annotations

import time

import pytest
import torch
import ttnn
from models.experimental.hunyuan_image_3_0.tt.parallel_utils import wide_mm_program_config

K, N = 4096, 6144  # hidden, 2 * intermediate (gate_and_up)
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
    """Explicit 2D-mcast on a gy x gx rectangular grid, K chunk = ibw."""
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
    """1D multicast, split N (=6144) across cores — natural for the Mt==1 (single
    token) shape where there is only one M-tile row to place."""
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


@pytest.mark.parametrize("S", [32, 64, 256], ids=["seq32_mt1", "seq64", "seq256"])
def test_expert_gateup_sweep(device, S):
    device.enable_program_cache()
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    M = ((S + _TILE - 1) // _TILE) * _TILE
    torch.manual_seed(0)
    x = torch.randn(1, S, K, dtype=torch.bfloat16) * 0.05
    w = torch.randn(K, N, dtype=torch.bfloat16) * 0.02  # [H, 2I] already transposed for ttnn.linear
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
    if Mt == 1:  # single-M-tile decode: 1D split-N is the natural schedule
        candidates.append(("1D split-N", _cfg_1d_split_n(grid, Mt, Kt, Nt)))
    # explicit 2D grids: vary gy (M rows) x gx (N cols) and the K chunk
    for gy in (Mt, max(1, Mt // 2), 1):
        for gx in (_div_leq(Nt, grid.x), _div_leq(Nt, max(1, grid.x // 2))):
            for ibw in (_div_leq(Kt, 4), _div_leq(Kt, 8)):
                if gy <= grid.y and gx <= grid.x:
                    c = _cfg_2d(grid, Mt, Kt, Nt, gy, gx, ibw)
                    if c:
                        candidates.append((f"2D gy={gy} gx={gx} ibw={ibw}", c))

    # Profiler hint for this op: in0_block_w=2, out_subblock 2x2. That needs
    # per_core_M % 2 == 0 and per_core_N % 2 == 0, i.e. Mt/gy and Nt/gx both even.
    # Enumerate grids that yield an even per-core M and N and force ibw=2 / 2x2.
    for gy in {g for g in (1, max(1, Mt // 2), Mt) if g <= grid.y and Mt % g == 0 and (Mt // g) % 2 == 0}:
        for gx in {g for g in (_div_leq(Nt, grid.x), _div_leq(Nt, max(1, grid.x // 2))) if (Nt // g) % 2 == 0}:
            pcm, pcn = Mt // gy, Nt // gx
            hint = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=2,
                out_subblock_h=2,
                out_subblock_w=2,
                per_core_M=pcm,
                per_core_N=pcn,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=False,
            )
            candidates.append((f"HINT gy={gy} gx={gx} ibw=2 osb=2x2", hint))

    seen, base = set(), None
    print(f"\n=== expert gate_up {M} x {K} x {N} (S={S}) ===")
    for name, pc in candidates:
        key = name if pc is None else (pc.compute_with_storage_grid_size, pc.in0_block_w, pc.per_core_M, pc.per_core_N)
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

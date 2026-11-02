# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-chip micro sweep of the prefill matmuls (cores-for-prefill study, dsv4-logs/cores/CORES_PF.md).

Shapes are the per-chip prefill projections at M = 1024 tokens (Up=2 x C=512) with the shipped numerics: bf16 activations, bfp8 weights, HiFi2, fp32 dest acc, packer L1 acc
(pf_tune.lin_ckc defaults). Each config runs REPS back-to-back eager launches with one sync; the time per launch is wall clock / REPS (device bound for the ops here, 80+ us).
Candidates: ttnn.experimental.minimal_matmul with the tt_dit heuristic blocking and neighbours on the 8x10 (shipped), 11x10 (tt_dit's Blackhole Galaxy clamp) and 12x10 grids.
Env: CMM_SHAPES (names csv), CMM_REPS, CMM_GRIDS ("8x10,11x10,12x10"), CMM_FULL=1 sweeps the block sizes, else heuristic + shipped only."""
import itertools
import math
import os
import time

import pytest
import torch

import ttnn

SHAPES = {  # name: (M, K, N) per chip
    "wqkv": (1024, 5120, 1792),
    "wq_b": (1024, 1280, 4096),
    "wo_a": (1024, 4608, 1024),
    "wo_b": (1024, 1024, 5120),
}
REPS = int(os.environ.get("CMM_REPS", "20"))


def heur(M, K, N, gx, gy):
    M_t, K_t, N_t = M // 32, K // 32, N // 32
    m_raw = max(1, math.ceil(M_t / gx))
    n_raw = N_t / gy
    if N_t <= 4:
        sb = (4, 1)
    elif N_t % 4 == 0:
        sb = (1, 4)
    elif N_t % 3 == 0 and N_t <= 24:
        sb = (1, 3)
    elif N_t % 2 == 0:
        sb = (1, 2)
    else:
        sb = (1, 1)
    Mb = min(max(sb[0], math.ceil(m_raw / sb[0]) * sb[0]), 16)
    Nb = min(max(sb[1], math.ceil(n_raw / sb[1]) * sb[1]), 16)
    while N_t % Nb != 0 and Nb > sb[1]:
        Nb -= sb[1]
    Nb = max(sb[1], Nb)
    Kb = min(8, K_t)
    while K_t % Kb != 0 and Kb > 1:
        Kb -= 1
    return Mb, Kb, Nb, sb


def timeit(fn):
    out = fn()
    ttnn.synchronize_device(out.device())
    ttnn.deallocate(out)
    t = time.perf_counter()
    for _ in range(REPS):
        o = fn()
        ttnn.deallocate(o)
    ttnn.synchronize_device(fn.dev)
    return (time.perf_counter() - t) / REPS * 1e6


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_cores_pf_mm(device):
    dev = device
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    names = os.environ.get("CMM_SHAPES", ",".join(SHAPES)).split(",")
    grids = [tuple(int(v) for v in g.split("x")) for g in os.environ.get("CMM_GRIDS", "8x10,11x10,12x10").split(",")]
    full = os.environ.get("CMM_FULL") == "1"
    g = torch.Generator().manual_seed(0)
    for nm in names:
        M, K, N = SHAPES[nm]
        x = torch.randn(1, 1, M, K, generator=g) * 0.5
        w = torch.randn(1, 1, K, N, generator=g) * 0.05
        ref = x @ w
        tx = ttnn.from_torch(
            x, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tw = ttnn.from_torch(
            w, device=dev, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        base = None
        cands = []
        for gx, gy in grids:
            hb = heur(M, K, N, gx, gy)
            cands.append(((gx, gy), hb[:3], hb[3]))
            if full:
                M_t, K_t, N_t = M // 32, K // 32, N // 32
                for mb, kb, nb in itertools.product((2, 3, 4, 8), (8, 16), (4, 8, 16)):
                    for sb in ((1, 4), (2, 2), (4, 1)):
                        if mb % sb[0] or nb % sb[1] or K_t % kb or N_t % nb:
                            continue
                        cands.append(((gx, gy), (mb, kb, nb), sb))
        cands.append(((8, 10), (8, 8, 8), None))
        seen, rows = set(), []
        for grid, blk, sb in cands:
            key = (grid, blk, sb)
            if key in seen:
                continue
            seen.add(key)
            kw = {}
            if sb:
                kw["subblock_h"], kw["subblock_w"] = sb
            try:
                cfg = ttnn.MinimalMatmulConfig(
                    M_block_size=blk[0],
                    K_block_size=blk[1],
                    N_block_size=blk[2],
                    compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
                    **kw,
                )

                def fn():
                    return ttnn.experimental.minimal_matmul(tx, tw, compute_kernel_config=ckc, config=cfg)

                fn.dev = dev
                us = timeit(fn)
                o = ttnn.to_torch(fn()).float()
                d = (o - ref).abs().max().item()
                pcc = torch.corrcoef(torch.stack([o.flatten(), ref.flatten()]))[0, 1].item()
                rows.append((us, grid, blk, sb, d, pcc))
            except Exception as e:  # noqa
                rows.append((1e9, grid, blk, sb, -1, str(e)[:60]))
        rows.sort(key=lambda r: r[0])
        sh = [r for r in rows if r[1] == (8, 10) and r[2] == (8, 8, 8) and r[3] is None]
        print(f"CMM {nm} M{M} K{K} N{N}: shipped {sh[0][0]:.1f} us" if sh else f"CMM {nm}: no shipped row", flush=True)
        for r in rows[:8]:
            print(
                f"CMM   {nm} {r[0]:8.1f} us grid {r[1]} blk {r[2]} sub {r[3]} maxdiff {r[4]:.4f} pcc {r[5]}", flush=True
            )
        ttnn.deallocate(tx)
        ttnn.deallocate(tw)

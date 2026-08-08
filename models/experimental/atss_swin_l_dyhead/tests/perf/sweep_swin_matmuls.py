# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Focused sweep for the top Stage-2 Swin-L matmuls (which dominate the steady-state
matmul bucket at the current 3.82 FPS baseline).

Profiled hot Stage-2 matmuls (HiFi2, 18 calls/iter each):
    S2_MLPwi   M=64,  K=768,  N=3072   GELU activation     9.5 ms / iter
    S2_MLPwo   M=64,  K=3072, N=768    no activation       5.4 ms / iter

M_tiles = 2  (M=64 -> only 2 tile rows). Default 8x8 routing leaves 75% of cores
idle in the M direction, so 1D-mcast and smaller-grid 2D-mcast variants are the
most promising.

Sweeps four candidate families:
    default       : ttnn.linear with auto-routing + core_grid=8x8 (production)
    2D mcast      : MatmulMultiCoreReuseMultiCastProgramConfig
    1D mcast      : MatmulMultiCoreReuseMultiCast1DProgramConfig (both mcast_in0)
    minimal       : ttnn.experimental.minimal_matmul + MinimalMatmulConfig

Run:
    bash /Users/gtobar/swin_optimization_autoresearch/run_bench.sh  # to re-baseline
    ssh cust-models-01 'cd /localdev/gtobar/swin_optimization && source local_env.sh && \
        cd tt-metal && pytest \
        models/experimental/atss_swin_l_dyhead/tests/perf/sweep_swin_matmuls.py -sv'
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn


TILE = 32
REPEATS = int(os.environ.get("SWIN_MM_REPEATS", "4"))


@dataclass(frozen=True)
class Shape:
    name: str
    M: int
    K: int
    N: int
    fidelity: ttnn.MathFidelity
    activation: str | None = None  # "gelu" or None


# IMPORTANT: M in the profile reads as 64[40] but that's only one leading dim of the
# input; ttnn.linear collapses leading dims so effective M = H*W. For Stage 2 input
# (1, 40, 40, 768) -> effective M=1600. Sweep below uses the true effective M.
SHAPES = [
    # Stage 2 (18 blocks, dim=768, input (1,40,40,768)) -- biggest leverage:
    Shape("S2_MLPwi", M=1600, K=768, N=3072, fidelity=ttnn.MathFidelity.HiFi2, activation="gelu"),
    Shape("S2_MLPwo", M=1600, K=3072, N=768, fidelity=ttnn.MathFidelity.HiFi2),
    Shape("S2_QKV", M=2304, K=768, N=2304, fidelity=ttnn.MathFidelity.HiFi2),
    Shape("S2_proj", M=2304, K=768, N=768, fidelity=ttnn.MathFidelity.LoFi),
    # Stage 3 (2 blocks, dim=1536, input (1,20,20,1536)) M=400:
    Shape("S3_MLPwi", M=400, K=1536, N=6144, fidelity=ttnn.MathFidelity.HiFi2, activation="gelu"),
    Shape("S3_MLPwo", M=400, K=6144, N=1536, fidelity=ttnn.MathFidelity.HiFi2),
]

# Grids worth testing. For large M (M_tiles >= 16) the full 8x8 grid is best;
# 1D variants with smaller grids are kept for small-M shapes (proj/QKV at S3).
GRIDS = [(8, 8), (8, 4), (4, 8), (4, 4), (8, 2), (2, 8)]


def _make_inputs(device, shape: Shape):
    torch.manual_seed(0)
    a = (torch.randn(1, 1, shape.M, shape.K) * 0.05).to(torch.bfloat16)
    w = (torch.randn(1, 1, shape.K, shape.N) * 0.05).to(torch.bfloat16)
    b = (torch.randn(1, 1, 1, shape.N) * 0.05).to(torch.bfloat16)
    a_dev = ttnn.from_torch(
        a,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w_dev = ttnn.from_torch(
        w,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b_dev = ttnn.from_torch(
        b,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return a_dev, w_dev, b_dev


def _time_call(fn, device):
    # 1 warmup + REPEATS timed back-to-back.
    out = fn()
    if out is not None:
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        out = fn()
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    return (t1 - t0) * 1e6 / REPEATS


def _run_default(a, w, b, shape, ck):
    return ttnn.linear(
        a,
        w,
        bias=b,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=ck,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        activation=shape.activation,
    )


def _run_2d(a, w, b, shape, ck, *, gx, gy, ibw, osh, osw):
    M_tiles = (shape.M + TILE - 1) // TILE
    N_tiles = shape.N // TILE
    pc_M = (M_tiles + gy - 1) // gy
    pc_N = (N_tiles + gx - 1) // gx
    fused_act = (ttnn.UnaryOpType.GELU, True) if shape.activation == "gelu" else None
    pcfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=ibw,
        out_subblock_h=osh,
        out_subblock_w=osw,
        per_core_M=pc_M,
        per_core_N=pc_N,
        transpose_mcast=False,
        fused_activation=fused_act,
    )
    return ttnn.linear(
        a,
        w,
        bias=b,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=ck,
        program_config=pcfg,
    )


def _run_1d(a, w, b, shape, ck, *, gx, gy, ibw, osh, osw, mcast_in0):
    M_tiles = (shape.M + TILE - 1) // TILE
    N_tiles = shape.N // TILE
    ncores = gx * gy
    pc_M = (M_tiles + ncores - 1) // ncores
    pc_N = N_tiles
    fused_act = (ttnn.UnaryOpType.GELU, True) if shape.activation == "gelu" else None
    pcfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=ibw,
        out_subblock_h=osh,
        out_subblock_w=osw,
        per_core_M=pc_M,
        per_core_N=pc_N,
        fuse_batch=True,
        fused_activation=fused_act,
        mcast_in0=mcast_in0,
    )
    return ttnn.linear(
        a,
        w,
        bias=b,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=ck,
        program_config=pcfg,
    )


def _run_minimal(a, w, b, shape, ck, *, gx, gy, M_block, K_block, N_block, sh, sw):
    fused_act = (ttnn.UnaryOpType.GELU, True) if shape.activation == "gelu" else None
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=M_block,
        K_block_size=K_block,
        N_block_size=N_block,
        subblock_h=sh,
        subblock_w=sw,
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
    )
    return ttnn.experimental.minimal_matmul(
        input_tensor=a,
        weight_tensor=w,
        bias_tensor=b,
        fused_activation=fused_act,
        config=cfg,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=ck,
    )


def _enum_variants(shape: Shape):
    """Yield (name, fn(a,w,b,ck)) — only configs that pass the divisibility/cap rules."""
    K_tiles = shape.K // TILE
    N_tiles = shape.N // TILE
    M_tiles = (shape.M + TILE - 1) // TILE

    yield ("default", lambda a, w, b, ck: _run_default(a, w, b, shape, ck))

    # 2D mcast
    for gx, gy in GRIDS:
        if N_tiles < gx:
            continue  # 2D mcast wants N_tiles >= gx
        pc_M = (M_tiles + gy - 1) // gy
        pc_N = (N_tiles + gx - 1) // gx
        for ibw in [1, 2, 4, 8]:
            if K_tiles % ibw != 0:
                continue
            for osh in [1, 2, 4]:
                if pc_M % osh != 0:
                    continue
                for osw in [1, 2, 4]:
                    if osw > pc_N or pc_N % osw != 0:
                        continue
                    if osh * osw > 8:
                        continue
                    name = f"2D_g{gx}x{gy}_ibw{ibw}_sub{osh}x{osw}"

                    def make(gx=gx, gy=gy, ibw=ibw, osh=osh, osw=osw):
                        return lambda a, w, b, ck: _run_2d(a, w, b, shape, ck, gx=gx, gy=gy, ibw=ibw, osh=osh, osw=osw)

                    yield (name, make())

    # 1D mcast (both mcast_in0 settings). Only useful when M_tiles or N_tiles is small
    # relative to a single dim of the grid -- skip for large-M shapes where the per_core_M
    # would otherwise be tiny over 64 cores.
    if M_tiles <= 16 or N_tiles <= 16:
        for gx, gy in GRIDS:
            ncores = gx * gy
            pc_M = (M_tiles + ncores - 1) // ncores
            pc_N = N_tiles
            for ibw in [1, 2, 4, 8]:
                if K_tiles % ibw != 0:
                    continue
                for osh in [1, 2, 4]:
                    if pc_M % osh != 0:
                        continue
                    for osw in [1, 2, 4]:
                        if osw > pc_N or pc_N % osw != 0:
                            continue
                        if osh * osw > 8:
                            continue
                        for mc0 in (False, True):
                            tag = "mc0" if mc0 else "mc1"
                            name = f"1D_g{gx}x{gy}_ibw{ibw}_sub{osh}x{osw}_{tag}"

                            def make(gx=gx, gy=gy, ibw=ibw, osh=osh, osw=osw, mc0=mc0):
                                return lambda a, w, b, ck: _run_1d(
                                    a, w, b, shape, ck, gx=gx, gy=gy, ibw=ibw, osh=osh, osw=osw, mcast_in0=mc0
                                )

                            yield (name, make())

    # minimal_matmul — anchor to (8, 8) grid only and tie M_block/N_block to
    # per_core_M/N to keep variant count manageable.
    for gx, gy in [(8, 8)]:
        if M_tiles < gy or N_tiles < gx:
            continue
        per_core_M = (M_tiles + gy - 1) // gy
        per_core_N = (N_tiles + gx - 1) // gx
        for M_block in [b for b in (1, 2, 4, 8) if per_core_M % b == 0]:
            for N_block in [b for b in (1, 2, 4, 8) if per_core_N % b == 0]:
                for K_block in [b for b in (4, 8) if K_tiles % b == 0]:
                    for sh in [b for b in (1, 2, 4) if M_block % b == 0]:
                        for sw in [b for b in (1, 2, 4) if N_block % b == 0]:
                            if sh * sw > 8:
                                continue
                            name = f"min_g{gx}x{gy}_M{M_block}K{K_block}N{N_block}_s{sh}x{sw}"

                            def make(gx=gx, gy=gy, Mb=M_block, Kb=K_block, Nb=N_block, sh=sh, sw=sw):
                                return lambda a, w, b, ck: _run_minimal(
                                    a, w, b, shape, ck, gx=gx, gy=gy, M_block=Mb, K_block=Kb, N_block=Nb, sh=sh, sw=sw
                                )

                            yield (name, make())


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.name)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_sweep_swin_matmul(device, shape):
    a, w, b = _make_inputs(device, shape)
    ck = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=shape.fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    out_dir = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated", "swin_mm_sweep")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"sweep_{shape.name}.csv")
    f_csv = open(csv_path, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["shape", "variant", "us_per_call", "status"])

    n_ok = 0
    n_err = 0
    for name, fn in _enum_variants(shape):
        try:
            t = _time_call(lambda: fn(a, w, b, ck), device)
            writer.writerow([shape.name, name, f"{t:.1f}", "ok"])
            n_ok += 1
        except Exception as e:
            writer.writerow([shape.name, name, "", f"ERR:{str(e)[:60]}"])
            n_err += 1
        f_csv.flush()

    logger.info(f"[{shape.name}] {n_ok} ok / {n_err} err. CSV={csv_path}")
    f_csv.close()
    ttnn.deallocate(a)
    ttnn.deallocate(w)
    ttnn.deallocate(b)

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Trace-timed program-config sweep for the encoder linears at one L1 chunk (64 series x 160 tokens).

Inputs, weights, dtypes and fidelities match TtChronosPrecision.performance(); a bf16 residual
of the chunk's size stays allocated in L1 next to each matmul, as it does in the model.
Families: the model's current config, 2D mcast variants (subblock, in0_block_w, out_block_h),
2D with M over grid columns (transpose_mcast), 1D in1-mcast and minimal_matmul. The fused
residual add of dit_minimal_matmul_addcmul_fused is not swept: it requires the residual in the
weight dtype (bf8), while the residual stream is bf16.

    python models/experimental/chronos_forecast/sweeps/sweep_matmul_l1.py qkv wo vo ff_up ff_down [--base]

``--base`` times only the config program_configs.linear picks.
"""

import itertools
import math
import sys
import time

import torch
import ttnn

from models.experimental.chronos_forecast.tt import program_configs

SERIES, T, D = 64, 133, 768
REPS = 10
GRID = (11, 10)

# name: (K, N, in0 is bf16, fidelity, relu)
SHAPES = {
    "qkv": (768, 2304, True, "HiFi2", False),
    "wo": (768, 768, False, "HiFi2", False),
    "vo": (768, 768, True, "HiFi2", False),
    "ff_up": (768, 3072, True, "LoFi", True),
    "ff_down": (3072, 768, False, "LoFi", False),
}


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def trace_us(dev, fn):
    out = fn()
    ttnn.deallocate(out)
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(REPS):
        ttnn.deallocate(fn())
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(3):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    us = (time.perf_counter() - t0) / 3 / REPS * 1e6
    ttnn.release_trace(dev, tid)
    return us


def subblocks(bh, bw, min_area=3):
    out = [(h, w) for h, w in itertools.product(range(1, 9), range(1, 9)) if h * w <= 8 and not bh % h and not bw % w]
    out = [s for s in out if s[0] * s[1] >= min(min_area, max(a * b for a, b in out))]
    return sorted(out, key=lambda s: -s[0] * s[1])


def divisors(n, lo, hi):
    return [d for d in range(lo, min(n, hi) + 1) if n % d == 0]


def configs(Mt, Kt, Nt, relu):
    act = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if relu else None
    gx, gy = GRID

    pcn = math.ceil(Nt / gx)
    pcm = math.ceil(Mt / gy)
    for obh in divisors(pcm, 8, 32):
        for ibw in divisors(Kt, 2, 12):
            for sh, sw in subblocks(obh, pcn):
                yield f"2d pcm{pcm} pcn{pcn} obh{obh} ibw{ibw} sb{sh}x{sw}", "mm", ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=GRID,
                    in0_block_w=ibw,
                    out_subblock_h=sh,
                    out_subblock_w=sw,
                    out_block_h=obh,
                    out_block_w=pcn,
                    per_core_M=pcm,
                    per_core_N=pcn,
                    transpose_mcast=False,
                    fused_activation=act,
                    fuse_batch=True,
                )

    if math.ceil(Nt / gx) * gx - Nt >= pcn:
        # N does not fill every grid column: put M over the 11 columns and N over the rows.
        tpcm = math.ceil(Mt / gx)
        tpcn = math.ceil(Nt / gy)
        while math.ceil(Nt / tpcn) > gy:
            tpcn += 1
        for obh in divisors(tpcm, 5, 30):
            for ibw in divisors(Kt, 2, 12):
                for sh, sw in subblocks(obh, tpcn):
                    yield f"2dT pcm{tpcm} pcn{tpcn} obh{obh} ibw{ibw} sb{sh}x{sw}", "mm", ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=GRID,
                        in0_block_w=ibw,
                        out_subblock_h=sh,
                        out_subblock_w=sw,
                        out_block_h=obh,
                        out_block_w=tpcn,
                        per_core_M=tpcm,
                        per_core_N=tpcn,
                        transpose_mcast=True,
                        fused_activation=act,
                        fuse_batch=True,
                    )

    for pcm1 in (3,):
        for obw in sorted({Nt, 24} & set(divisors(Nt, 1, Nt))):
            for ibw in divisors(Kt, 2, 8):
                for sh, sw in subblocks(pcm1, obw):
                    yield f"1d pcm{pcm1} obw{obw} ibw{ibw} sb{sh}x{sw}", "mm", ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=GRID,
                        in0_block_w=ibw,
                        out_subblock_h=sh,
                        out_subblock_w=sw,
                        out_block_h=pcm1,
                        out_block_w=obw,
                        per_core_M=pcm1,
                        per_core_N=Nt,
                        fuse_batch=True,
                        fused_activation=act,
                        mcast_in0=False,
                    )

    n_blocks = {24: (2, 3, 4), 72: (4, 6, 8), 96: (4, 8)}[Nt]
    k_blocks = [k for k in (4, 6, 8, 12, 16) if not Kt % k]
    for mb, kb, nb in itertools.product((8, 16), k_blocks, n_blocks):
        for sh, sw in subblocks(mb, nb)[:1]:
            yield f"mmm m{mb} k{kb} n{nb} sb{sh}x{sw}", "mmm", ttnn.MinimalMatmulConfig(
                M_block_size=mb,
                K_block_size=kb,
                N_block_size=nb,
                subblock_h=sh,
                subblock_w=sw,
                compute_with_storage_grid_size=ttnn.CoreCoord(*GRID),
            )


def main(which, base_only=False):
    K, N, in0_bf16, fid_name, relu = SHAPES[which]
    fid = getattr(ttnn.MathFidelity, fid_name)
    ckc = program_configs.compute_kernel_config(fid)
    l1 = ttnn.L1_MEMORY_CONFIG
    Mt, Kt, Nt = SERIES * 160 // 32, K // 32, N // 32

    torch.manual_seed(0)
    x_h = torch.randn(SERIES, T, K)
    w_h = torch.randn(K, N) / K**0.5
    ref = x_h @ w_h
    if relu:
        ref = torch.relu(ref)

    in0_dtype = ttnn.bfloat16 if in0_bf16 else ttnn.bfloat8_b
    x = ttnn.from_torch(x_h, dtype=in0_dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=l1)
    w = ttnn.from_torch(w_h, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    ballast = ttnn.from_torch(
        torch.randn(SERIES, T, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=l1
    )
    act = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if relu else None

    def run(kind, cfg):
        if kind == "base":
            return lambda: program_configs.linear(
                x, w, activation="relu" if relu else None, memory_config=l1, dtype=ttnn.bfloat8_b, math_fidelity=fid
            )
        if kind == "mm":
            return lambda: ttnn.linear(
                x, w, program_config=cfg, memory_config=l1, dtype=ttnn.bfloat8_b, compute_kernel_config=ckc
            )
        if kind == "mmm":
            return lambda: ttnn.experimental.minimal_matmul(
                x,
                w,
                fused_activation=act,
                config=cfg,
                memory_config=l1,
                dtype=ttnn.bfloat8_b,
                compute_kernel_config=ckc,
            )
        raise ValueError(kind)

    results = []

    def measure(name, kind, cfg, want):
        try:
            fn = run(kind, cfg)
            out = fn()
            got = ttnn.to_torch(out).float()[:, :T, :N]
            ttnn.deallocate(out)
            p = pcc(want, got)
            if p < 0.99:
                print(f"[{which}] {name}: BAD pcc={p:.4f}", flush=True)
                return
            us = trace_us(dev, fn)
        except Exception as e:  # noqa: BLE001
            print(f"[{which}] {name}: ERR {str(e).splitlines()[0][:140]}", flush=True)
            return
        results.append((us, name))
        print(f"[{which}] {name}: {us:8.1f} us pcc={p:.5f}", flush=True)

    measure("base (model config)", "base", None, ref)
    for name, kind, cfg in [] if base_only else configs(Mt, Kt, Nt, relu):
        measure(name, kind, cfg, ref)

    results.sort()
    print(f"\n[{which}] best:", flush=True)
    for us, name in results[:10]:
        print(f"   {us:8.1f} us  {name}", flush=True)
    for t in (x, w, ballast):
        ttnn.deallocate(t)


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0, trace_region_size=50_000_000)
    dev.enable_program_cache()
    args = [a for a in sys.argv[1:] if a != "--base"]
    try:
        for which in args or list(SHAPES):
            main(which, base_only="--base" in sys.argv)
    finally:
        ttnn.close_device(dev)

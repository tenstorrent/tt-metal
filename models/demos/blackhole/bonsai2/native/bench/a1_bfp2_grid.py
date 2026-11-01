# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""bfp2 vs bfp4 with the grid swept too, at the best block width.

Established: bfp2 and bfp4 produce BIT-IDENTICAL output (512/512, max diff 0.0), so
correctness was never the issue and the earlier PCC 0.883 was a broken comparison
(a scaled reference against a device that applies no scales).

That leaves speed, and the block sweep says something specific. Raising
in0_block_w from 4 to 32 makes bfp2 *slower* (183.4 -> 170.0 us) while bfp4 goes
183.3 -> 185.9, so per_core_N=2 does not divide 544/8 tiles into a fast shape, and
at in0_block_w=8, where per_core_N becomes 3 and 26 cores idle, bfp2 is 1.08x over
bfp4 rather than the 1.5x the bytes imply.

per_core_N is chosen as ceil(n_tiles / cores) and both shapes have n_tiles=160, so
the grid and the block count are coupled. This decouples them: for each in0_block_w
it sweeps the core count so that 160 output tiles divide evenly, giving per_core_N
1, 2, 4, 5, 8 and no idle cores, and reports bfp2/bfp4 at each.

The question it answers: is the 1.08x a property of the bytes, or of the tiling the
current config forces?

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a1_bfp2_grid
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch

from .a1_calib import device_kernel_us
from .placement import act_mem, describe, out_mem
from .record import DRAM_PEAK_GBS, append_jsonl, new_record
from .timing import Timer


def weight_mb(n: int, k: int, fmt: str) -> float:
    if fmt == "bfp8":
        return (n * k + n * k // 16) / 1e6
    if fmt == "bfp4":
        return (n * k // 2 + n * k // 16) / 1e6
    if fmt == "bfp2":
        return (n * k // 4 + n * k // 16) / 1e6
    raise ValueError(fmt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="ffn_down")
    ap.add_argument("--k", type=int, default=17408)
    ap.add_argument("--n", type=int, default=5120)
    ap.add_argument("--bws", default="4,8")
    ap.add_argument("--fmts", default="bfp2,bfp4")
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=9)
    ap.add_argument("--out", default=os.path.expanduser("~/avan/native2bit/agentA/logs/a1_bfp2_grid.jsonl"))
    args = ap.parse_args()

    for key, val in (("TT_METAL_DEVICE_PROFILER", "1"),
                     ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
                     ("TT_METAL_PROFILER_MID_RUN_DUMP", "1")):
        os.environ.setdefault(key, val)

    import ttnn

    k, n = args.k, args.n
    k_tiles, n_tiles = k // 32, n // 32
    name = args.shape
    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    timer = Timer(warmup=3, calls=args.calls, repeat=args.repeat)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True)
    codes = np.random.default_rng(0).integers(-1, 2, (n, k)).astype(np.float32)
    dtypes = {"bfp2": ttnn.DataType.BFLOAT2_B, "bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}

    # Core counts that divide 160 output tiles evenly, plus the shapes a grid of
    # that size can actually take (cols<=11, rows<=10).
    grids = []
    for pcN in (1, 2, 4, 5, 8):
        cores = n_tiles // pcN
        cols, rows = min(11, cores), math.ceil(cores / min(11, cores))
        if cols * rows <= 110 and cores <= 110:
            grids.append((cores, pcN, cols, rows))
    print(f"{name} K={k} N={n} ({k_tiles} K-tiles, {n_tiles} N-tiles), {describe(name)}")
    print(f"grids (cores, per_core_N, cols x rows): {[(g[0], g[1], f'{g[2]}x{g[3]}') for g in grids]}")
    print(f"{'fmt':5s} {'in0bw':>5} {'cores':>5} {'pcN':>4} {'MB':>8} {'dev us':>8} "
          f"{'GB/s':>7} {'%pk':>5}")
    rc = 0
    try:
        x = ttnn.from_torch(
            torch.randn((1, 1, 1, k), dtype=torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=act_mem(ttnn, name))
        res = {}
        for fmt in args.fmts.split(","):
            wt = ttnn.from_torch(torch.from_numpy(codes.T.copy()), dtype=dtypes[fmt],
                                 layout=ttnn.TILE_LAYOUT, device=device,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
            mb = weight_mb(n, k, fmt)
            for bw in [int(b) for b in args.bws.split(",")]:
                if k_tiles % bw:
                    continue
                for cores, pcN, cols, rows in grids:
                    subW = max([i for i in range(1, 5) if pcN % i == 0] or [1])
                    try:
                        progcfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                            compute_with_storage_grid_size=(cols, rows),
                            in0_block_w=bw, out_subblock_h=1, out_subblock_w=subW,
                            per_core_M=1, per_core_N=pcN, fuse_batch=True,
                            fused_activation=None, mcast_in0=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"{fmt:5s} {bw:5d} {cores:5d} progcfg FAILED {str(e)[:50]}")
                        continue

                    def once(wt=wt, progcfg=progcfg):
                        y = ttnn.linear(x, wt, compute_kernel_config=ckc,
                                        program_config=progcfg,
                                        memory_config=out_mem(ttnn, name))
                        ttnn.synchronize_device(device)
                        ttnn.deallocate(y)

                    try:
                        med, samples = timer.best(once)
                        once()
                        dev = device_kernel_us(ttnn, device) or med * 1e6
                    except Exception as e:  # noqa: BLE001
                        print(f"{fmt:5s} {bw:5d} {cores:5d} FAILED {type(e).__name__}: "
                              f"{str(e)[:70]}", flush=True)
                        rc = 1
                        continue
                    gbs = mb / (dev * 1e-6) / 1e9
                    print(f"{fmt:5s} {bw:5d} {cores:5d} {pcN:4d} {mb:8.2f} {dev:8.1f} "
                          f"{gbs:7.1f} {gbs / DRAM_PEAK_GBS * 100:4.1f}%", flush=True)
                    res[(fmt, bw, cores)] = dev
                    rec = new_record(f"grid_{name}", K=k, N=n, fmt=fmt, cores=cores,
                                     in0_block_w=bw, per_core_N=pcN, placement=describe(name))
                    rec.samples = samples
                    append_jsonl(args.out, rec, int(mb * 1e6))
            ttnn.deallocate(wt)
        print()
        print("bfp2/bfp4 ratio at each (in0_block_w, cores):")
        for bw in [int(b) for b in args.bws.split(",")]:
            for cores, pcN, _, _ in grids:
                a, b = res.get(("bfp2", bw, cores)), res.get(("bfp4", bw, cores))
                if a and b:
                    print(f"  in0bw={bw:2d} cores={cores:3d} pcN={pcN}: "
                          f"bfp2 {a:6.1f} bfp4 {b:6.1f} -> {b / a:.2f}x")
    finally:
        ttnn.close_device(device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

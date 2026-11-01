# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-block floor vs 4-bit traffic: the in0_block_w sweep that tells them apart.

bfp2 and bfp4 time identically (183.4 us) on ffn_down at in0_block_w=4, and the
planner's arithmetic explains why that cannot distinguish two very different causes.

With per_core_N=2, 80 of 110 cores work, each over 544/4 = 136 K-blocks of 8 in1
tiles, so 183.4 us is 1.35 us per block for *both* formats (2560 B per block for
bfp2, 4608 B for bfp4). Two hypotheses predict that exactly:

  per-block floor   a fixed cost per block (the in1 read round trip plus the in0
                    mcast handshake) dominates, so every format whose blocks fit
                    the bandwidth at that pace takes the same time. bfp8 is slower
                    only because its 8.7 KB blocks would need 516 GB/s to keep up.
  4-bit traffic     the device is moving 4 bits per value for bf2 as well, so its
                    blocks are really the same size as bf4's.

The sweep separates them. Raising in0_block_w amortises a per-block floor, so under
that hypothesis both formats speed up AND bfp2 pulls ahead of bfp4. Under 4-bit
traffic bfp2 keeps tracking bfp4 at every block width, because they are moving the
same bytes. This would be the in0_block_w=2 trap one level down.

Valid block widths for ffn_down are the divisors of k_tiles=544 that also satisfy
the matmul's constraints: 4, 8, 16, 32.

Everything is device-kernel-duration, on the model's own placement, with the
corrected x @ w orientation (bench/a1_orient.py).

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a1_bfp2_blocks
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
    ap.add_argument("--cores", default="110")
    ap.add_argument("--bws", default="4,8,16,32")
    ap.add_argument("--fmts", default="bfp2,bfp4,bfp8")
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=9)
    ap.add_argument("--out", default=os.path.expanduser("~/avan/native2bit/agentA/logs/a1_bfp2_blocks.jsonl"))
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
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    # Ternary codes: every format can store these exactly, so any timing or value
    # difference is about the format's traffic and unpacking, not about rounding.
    codes = np.random.default_rng(0).integers(-1, 2, (n, k)).astype(np.float32)

    dtypes = {"bfp2": ttnn.DataType.BFLOAT2_B, "bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}
    print(f"{name} K={k} N={n} ({k_tiles} K-tiles), {describe(name)}, "
          f"peak {DRAM_PEAK_GBS} GB/s, device kernel duration")
    print(f"{'fmt':5s} {'in0bw':>5} {'cores':>5} {'pcN':>4} {'MB':>8} {'dev us':>8} "
          f"{'GB/s':>7} {'%pk':>5} {'us/block':>9} {'B/block':>8}")
    rc = 0
    try:
        x = ttnn.from_torch(
            torch.randn((1, 1, 1, k), dtype=torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=act_mem(ttnn, name))
        results = {}
        for fmt in args.fmts.split(","):
            wt = ttnn.from_torch(
                torch.from_numpy(codes.T.copy()), dtype=dtypes[fmt],
                layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            mb = weight_mb(n, k, fmt)
            for cores in [int(c) for c in args.cores.split(",")]:
                for bw in [int(b) for b in args.bws.split(",")]:
                    if k_tiles % bw:
                        continue
                    pcN = math.ceil(n_tiles / cores)
                    cols, rows = min(11, cores), math.ceil(cores / min(11, cores))
                    subW = max([i for i in range(1, 5) if pcN % i == 0] or [1])
                    try:
                        progcfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                            compute_with_storage_grid_size=(cols, rows),
                            in0_block_w=bw, out_subblock_h=1, out_subblock_w=subW,
                            per_core_M=1, per_core_N=pcN, fuse_batch=True,
                            fused_activation=None, mcast_in0=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"{fmt:5s} {bw:5d} progcfg FAILED {str(e)[:60]}", flush=True)
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
                        print(f"{fmt:5s} {bw:5d} FAILED {type(e).__name__}: {str(e)[:90]}",
                              flush=True)
                        rc = 1
                        continue
                    k_blocks = k_tiles // bw
                    # Bytes each *worker* pulls per block: bw in1 tiles of this
                    # format's tile size, which is what the floor hypothesis is about.
                    tile_bytes = {"bfp2": 320, "bfp4": 576, "bfp8": 1088}[fmt]
                    per_block_bytes = bw * tile_bytes
                    gbs = mb / (dev * 1e-6) / 1e9
                    print(f"{fmt:5s} {bw:5d} {cores:5d} {pcN:4d} {mb:8.2f} {dev:8.1f} "
                          f"{gbs:7.1f} {gbs / DRAM_PEAK_GBS * 100:4.1f}% "
                          f"{dev / k_blocks:9.3f} {per_block_bytes:8d}", flush=True)
                    results[(fmt, cores, bw)] = (dev, mb, per_block_bytes, k_blocks)
                    rec = new_record(f"blocks_{name}", K=k, N=n, fmt=fmt, cores=cores,
                                     in0_block_w=bw, placement=describe(name))
                    rec.samples = samples
                    append_jsonl(args.out, rec, int(mb * 1e6))
            ttnn.deallocate(wt)

        # ---- the verdict, computed rather than asserted
        print()
        cores_used = int(args.cores.split(",")[0])
        for bw in [int(b) for b in args.bws.split(",")]:
            a = results.get(("bfp2", cores_used, bw))
            b = results.get(("bfp4", cores_used, bw))
            if not (a and b):
                continue
            ratio = b[0] / a[0]
            print(f"in0_block_w={bw:2d}: bfp2 {a[0]:6.1f} us vs bfp4 {b[0]:6.1f} us "
                  f"-> {ratio:.2f}x   (bfp2 {a[3]} blocks at {a[0] / a[3]:.3f} us/block, "
                  f"bfp4 {b[3]} at {b[0] / b[3]:.3f})")
        print()
        bw_lo = min(results.get(("bfp2", cores_used, int(b)), (1e9,))[0]
                    for b in args.bws.split(",") if ("bfp2", cores_used, int(b)) in results)
        bw_hi = max(results.get(("bfp2", cores_used, int(b)), (0,))[0]
                    for b in args.bws.split(",") if ("bfp2", cores_used, int(b)) in results)
        print("READ THIS:")
        if bw_hi < 0.85 * bw_lo:
            print("  bfp2 gets materially faster as in0_block_w rises, so the block cost")
            print("  has a per-block component that amortises. Check whether bfp2 pulls")
            print("  away from bfp4 at the same widths: if it does, no floor is implied by")
            print("  the bytes and a smaller format is the win; if both scale together, the")
            print("  floor is a property of this program config and not of the format.")
        else:
            print("  bfp2 barely moves with in0_block_w, so there is no large per-block")
            print("  floor to amortise on this shape. That points away from the floor")
            print("  hypothesis and back at the bytes actually being moved.")
    finally:
        ttnn.close_device(device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

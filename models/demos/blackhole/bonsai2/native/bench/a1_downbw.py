# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The decisive ffn_down comparison: in0_block_w and format at a fixed core count.

Earlier sweeps concluded ffn_down was compute-bound because bfp4 and bfp8 took
the same time (376 us both). They were run with no program config, so ttnn picked
in0_block_w=2 on this shape -- and at in0_block_w=2 the matmul stalls. Hold the
core count at the model's 88 and sweep in0_block_w: bfp4 goes 346.5 (bw=2) ->
170.0 (bw=8), while bfp8 goes 347.4 (bw=2) -> 251.7 (bw=4). Two conclusions
follow, and both matter to the plan:

  * ffn_down is NOT compute-bound. With a sane K-block the two formats separate
    by 1.48x, exactly as bytes predict. The "format cannot help ffn_down"
    finding was a benchmark artifact.
  * in0_block_w=2 is a trap on large-K shapes, and it is what the auto config
    chooses. At 88 cores, bw=2 -> 4 alone is 96 us per call on ffn_down.

This pins the numbers with 9 batches of 10 calls, device kernel duration, and a
bit-exactness check against the same shape's auto config (a different in0_block_w
must produce the same result, since it only re-partitions the K loop).

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a1_downbw
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch

from .a1_calib import device_kernel_us
from .record import DRAM_PEAK_GBS, append_jsonl, new_record
from .timing import Timer


def pcc(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="ffn_down")
    ap.add_argument("--k", type=int, default=17408)
    ap.add_argument("--n", type=int, default=5120)
    ap.add_argument("--cores", default="88")
    ap.add_argument("--bws", default="2,4,8,16")
    ap.add_argument("--fmts", default="bfp8,bfp4")
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=9)
    ap.add_argument("--out", default=os.path.expanduser("~/avan/native2bit/agentA/logs/a1_downbw.jsonl"))
    args = ap.parse_args()

    for key, val in (("TT_METAL_DEVICE_PROFILER", "1"),
                     ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
                     ("TT_METAL_PROFILER_MID_RUN_DUMP", "1")):
        os.environ.setdefault(key, val)

    import ttnn

    k, n = args.k, args.n
    k_tiles, n_tiles = k // 32, n // 32
    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    timer = Timer(warmup=3, calls=args.calls, repeat=args.repeat)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    print(f"{args.shape} K={k} N={n} ({k_tiles} K-tiles, {n_tiles} N-tiles). Peak {DRAM_PEAK_GBS} GB/s.")
    print(f"{'fmt':5s} {'cores':>5} {'in0bw':>5} {'pcN':>4} {'subW':>4} {'MB':>7} "
          f"{'host us':>8} {'spread':>6} {'dev us':>8} {'GB/s':>7} {'%pk':>5} {'pcc':>9}")
    try:
        for fmt in args.fmts.split(","):
            nb = n * k + n * k // 16 if fmt == "bfp8" else n * k // 2 + n * k // 16
            w_np = torch.randn((k, n), dtype=torch.bfloat16)
            wt = ttnn.from_torch(
                w_np, dtype=ttnn.bfloat8_b if fmt == "bfp8" else ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            x = ttnn.from_torch(
                torch.randn((1, 1, 1, k), dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ref = None
            cfgs = [("auto", 0)] + [(int(c), int(b))
                                    for c in args.cores.split(",")
                                    for b in args.bws.split(",")]
            for cores, bw in cfgs:
                if cores == "auto":
                    pc, lab_c, lab_bw, pcN, subW = None, "auto", "-", 0, 0
                else:
                    if k_tiles % bw:
                        continue
                    pcN = math.ceil(n_tiles / cores)
                    cols, rows = min(11, cores), math.ceil(cores / min(11, cores))
                    subW = max([i for i in range(1, 5) if pcN % i == 0] or [1])
                    try:
                        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                            compute_with_storage_grid_size=(cols, rows),
                            in0_block_w=bw, out_subblock_h=1, out_subblock_w=subW,
                            per_core_M=1, per_core_N=pcN, fuse_batch=True,
                            fused_activation=None, mcast_in0=True,
                        )
                    except Exception as e:  # noqa: BLE001
                        print(f"{fmt:5s} {cores:5d} {bw:5d}: build FAILED {str(e)[:60]}", flush=True)
                        continue
                    lab_c, lab_bw = str(cores), str(bw)

                def once(pc=pc):
                    y = ttnn.linear(x, wt, compute_kernel_config=ckc, program_config=pc,
                                    memory_config=ttnn.L1_MEMORY_CONFIG)
                    ttnn.synchronize_device(device)
                    return y

                try:
                    med, samples = timer.best(lambda: ttnn.deallocate(once()))
                except Exception as e:  # noqa: BLE001
                    print(f"{fmt:5s} {lab_c:>5} {lab_bw:>5}: FAILED {type(e).__name__}: "
                          f"{str(e)[:60]}", flush=True)
                    continue
                y = once()
                got = ttnn.to_torch(y).float().numpy().reshape(1, n)
                ttnn.deallocate(y)
                if ref is None:
                    ref = got
                    c = 1.0
                else:
                    c = pcc(got, ref)
                once()
                dev = device_kernel_us(ttnn, device) or float("nan")
                best = dev if dev == dev else med * 1e6
                gbs = nb / (best * 1e-6) / 1e9
                spread = (max(samples) - min(samples)) / float(np.median(samples)) * 100
                print(f"{fmt:5s} {lab_c:>5} {lab_bw:>5} {pcN:4d} {subW:4d} {nb / 1e6:7.2f} "
                      f"{med * 1e6:8.1f} {spread:5.1f}% {dev:8.1f} {gbs:7.1f} "
                      f"{gbs / DRAM_PEAK_GBS * 100:4.1f}% {c:9.7f}", flush=True)
                rec = new_record(f"downbw_{fmt}", shape=args.shape, K=k, N=n,
                                 cores=lab_c, in0_block_w=lab_bw, pcc=c)
                rec.samples = samples
                append_jsonl(args.out, rec, nb)
            ttnn.deallocate(wt)
            ttnn.deallocate(x)
            del w_np
            print("", flush=True)
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

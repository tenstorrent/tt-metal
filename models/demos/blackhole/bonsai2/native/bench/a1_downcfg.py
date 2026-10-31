# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ffn_down is the clearest floor case: is it math, or per-block sync?

The M sweep settles that it is not MVMUL work: bfp4 ffn_down takes 377.9 us at
M=1, 376.2 at M=32 and 406.5 at M=64 -- flat across a 64x range of MVMUL work.
So the ~315 us "compute floor" the earlier fit reported is not matrix math. It is
something per-tile: DRAM streaming at 53% of peak with a stall the config may be
able to remove, and the plan's own steer (11:33, item 2) points at exactly this.

ffn_down is the shape worth fixing: 64 calls per token, 22 ms/token of the dense
decode budget. Every config here is timed on DEVICE KERNEL DURATION (the host
median is ~30 us above it, which is why the earlier sweeps overstated the floor).

Swept, at bfp8 and bfp4:

  * core count: 11..110, the model uses 88 today
  * in0_block_w: the K-block chunk (2, 4, 8, 16 -- must divide k_tiles=544)
  * per_core_N and out_subblock: how many output tiles each core owns
  * mcast_in0 1D vs DRAM-sharded

Any row that beats 347.7 us (the model's current device time) is a direct win on
the dense model, independent of ternary work.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a1_downcfg
"""

from __future__ import annotations

import argparse
import math
import os

import torch

from .a1_calib import device_kernel_us
from .record import DRAM_PEAK_GBS, append_jsonl, new_record
from .timing import Timer


def divisors(x: int) -> list[int]:
    return [d for d in range(1, x + 1) if x % d == 0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=17408)
    ap.add_argument("--n", type=int, default=5120)
    ap.add_argument("--shape", default="ffn_down")
    ap.add_argument("--calls", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", default=os.path.expanduser("~/avan/native2bit/agentA/logs/a1_downcfg.jsonl"))
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
    nb = {"bfp8": n * k + n * k // 16, "bfp4": n * k // 2 + n * k // 16}

    print(f"{args.shape} K={k} N={n}: {k_tiles} K-tiles x {n_tiles} N-tiles. "
          f"Peak {DRAM_PEAK_GBS} GB/s. Model today: 347.7 us @ bfp8, 88 cores, 53%.")
    print(f"{'fmt':5s} {'cores':>6} {'grid':>7} {'in0bw':>6} {'pcN':>4} {'subW':>5} "
          f"{'kind':>6} {'host us':>8} {'dev us':>8} {'GB/s':>7} {'%pk':>5}")
    try:
        for fmt in ("bfp8", "bfp4"):
            wt = ttnn.from_torch(
                torch.randn((k, n), dtype=torch.bfloat16),
                dtype=ttnn.bfloat8_b if fmt == "bfp8" else ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            x = ttnn.from_torch(
                torch.randn((1, 1, 1, k), dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            def measure(progcfg, kind):
                try:
                    def once():
                        y = ttnn.linear(x, wt, compute_kernel_config=ckc, program_config=progcfg,
                                        memory_config=ttnn.L1_MEMORY_CONFIG)
                        ttnn.synchronize_device(device)
                        ttnn.deallocate(y)
                    med, samples = timer.best(once)
                    return med * 1e6, samples
                except Exception as e:  # noqa: BLE001
                    print(f"  {fmt} {kind}: FAILED {str(e)[:80]}", flush=True)
                    return None, None

            # ---- baseline: what the model does (no progcfg, auto 1D)
            dev_us = None
            med, samples = measure(None, "auto")
            if med:
                once_sync = ttnn.linear(x, wt, compute_kernel_config=ckc,
                                        memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.synchronize_device(device)
                ttnn.deallocate(once_sync)
                dev_us = device_kernel_us(ttnn, device)
                best = dev_us or med
                print(f"{fmt:5s} {'auto':>6} {'auto':>7} {'-':>6} {'-':>4} {'-':>5} "
                      f"{'auto':>6} {med:8.1f} {(dev_us or float('nan')):8.1f} "
                      f"{nb[fmt] / (best * 1e-6) / 1e9:7.1f} "
                      f"{nb[fmt] / (best * 1e-6) / 1e9 / DRAM_PEAK_GBS * 100:4.1f}%", flush=True)
                rec = new_record(f"downcfg_{fmt}", shape=args.shape, K=k, N=n, cfg="auto")
                rec.samples = samples
                append_jsonl(args.out, rec, nb[fmt])

            # ---- 1D mcast sweep over core count and in0_block_w
            for cores in (33, 44, 55, 66, 88, 99, 110):
                for in0bw in (2, 4, 8, 16, 32):
                    if k_tiles % in0bw:
                        continue
                    per_core_n = math.ceil(n_tiles / cores)
                    # Grid shaped wide-first, the way model_config does it.
                    cols = min(11, cores)
                    rows = math.ceil(cores / cols)
                    if cols * rows < n_tiles:
                        pass
                    cap = 4
                    sub_w = max([i for i in range(1, cap + 1) if per_core_n % i == 0] or [1])
                    try:
                        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                            compute_with_storage_grid_size=(cols, rows),
                            in0_block_w=in0bw, out_subblock_h=1, out_subblock_w=sub_w,
                            per_core_M=1, per_core_N=per_core_n, fuse_batch=True,
                            fused_activation=None, mcast_in0=True,
                        )
                    except Exception as e:  # noqa: BLE001
                        continue
                    med, samples = measure(pc, "1d")
                    if not med:
                        continue
                    once_sync = ttnn.linear(x, wt, compute_kernel_config=ckc, program_config=pc,
                                            memory_config=ttnn.L1_MEMORY_CONFIG)
                    ttnn.synchronize_device(device)
                    ttnn.deallocate(once_sync)
                    dev_us = device_kernel_us(ttnn, device)
                    best = dev_us or med
                    gbs = nb[fmt] / (best * 1e-6) / 1e9
                    print(f"{fmt:5s} {cores:6d} {cols}x{rows:<4d} {in0bw:6d} {per_core_n:4d} "
                          f"{sub_w:5d} {'1d':>6} {med:8.1f} {(dev_us or float('nan')):8.1f} "
                          f"{gbs:7.1f} {gbs / DRAM_PEAK_GBS * 100:4.1f}%", flush=True)
                    rec = new_record(f"downcfg_{fmt}", shape=args.shape, K=k, N=n,
                                     cores=cores, grid=f"{cols}x{rows}", in0_block_w=in0bw,
                                     per_core_N=per_core_n, kind="1d")
                    rec.samples = samples
                    append_jsonl(args.out, rec, nb[fmt])
            ttnn.deallocate(wt)
            ttnn.deallocate(x)
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

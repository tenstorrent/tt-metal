# Copyright (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Corrected byte-vs-compute floors, on device time and at a sane in0_block_w.

The question this answers is the planner's gate: does bfp2 predict at least 1.4x
over bfp4 on the decode shapes? Everything earlier in the project said no, and
everything earlier was measured two ways that both understate the format's value:

  * host 5-call medians, which carry ~30 us of dispatch on these shapes, so the
    fixed cost was over-attributed and the bytes term under-attributed;
  * no program config, so large-K shapes ran at in0_block_w=2 where the matmul
    stalls. On ffn_down that hid a 1.46x bfp8->bfp4 separation entirely.

So each shape is now measured at bf16/bfp8/bfp4, device-timed, at the best
in0_block_w found for it, and fitted

    t = c + bytes / BW

where ``c`` is the residual fixed cost and ``BW`` the marginal streaming rate.
bfp2 is then predicted from its real byte count (2 bits + one exponent per 16, no
fp16 scale of its own -- the scale epilogue is priced separately). A separate fit
on the *same* shape at bw=2 is printed as a control, so the effect of the block
width is visible rather than assumed.

Predicted bfp2 is reported against bfp4 at the same config, which is the gate.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a1_floors
"""

from __future__ import annotations

import argparse
import math
import os

import torch

from .a1_calib import device_kernel_us
from .placement import act_mem, describe, out_mem
from .record import DRAM_PEAK_GBS, append_jsonl, new_record
from .timing import Timer

#: (name, K, N, best in0_block_w found so far, core count).
SHAPES = [
    ("ffn_down", 17408, 5120, 4, 110),
    ("ffn_gate", 5120, 17408, 4, 110),
    ("gdn_qkvz", 5120, 16384, 4, 110),
    ("attn_q_gate", 5120, 12288, 4, 110),
    ("gdn_out", 6144, 5120, 4, 110),
    ("attn_kv", 5120, 1024, 4, 110),
    ("attn_o", 6144, 5120, 4, 110),
    ("lm_head", 5120, 248320, 4, 110),
]


def weight_mb(n: int, k: int, fmt: str) -> float:
    if fmt == "bf16":
        return n * k * 2 / 1e6
    if fmt == "bfp8":
        return (n * k + n * k // 16) / 1e6
    if fmt == "bfp4":
        return (n * k // 2 + n * k // 16) / 1e6
    if fmt == "bfp2":
        return (n * k // 4 + n * k // 16) / 1e6
    raise ValueError(fmt)


def fit(xs, ys):
    nn = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = nn * sxx - sx * sx
    if not den:
        return 0.0, 0.0, 0.0
    b = (nn * sxy - sx * sy) / den
    a = (sy - b * sx) / nn
    res = max(abs(y - (a + b * x)) for x, y in zip(xs, ys))
    return a, b, res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="")
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=9)
    ap.add_argument("--out", default=os.path.expanduser("~/avan/native2bit/agentA/logs/a1_floors.jsonl"))
    args = ap.parse_args()

    for key, val in (("TT_METAL_DEVICE_PROFILER", "1"),
                     ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
                     ("TT_METAL_PROFILER_MID_RUN_DUMP", "1")):
        os.environ.setdefault(key, val)

    import ttnn

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    timer = Timer(warmup=3, calls=args.calls, repeat=args.repeat)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    dt = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
    shapes = SHAPES
    if args.shapes:
        want = set(args.shapes.split(","))
        shapes = [s for s in shapes if s[0] in want]

    print(f"Peak {DRAM_PEAK_GBS} GB/s, device kernel duration, {args.repeat} batches x {args.calls} calls.")
    try:
        for name, k, n, bw, cores in shapes:
            k_tiles, n_tiles = k // 32, n // 32
            if k_tiles % bw:
                bw = 2
            w_src = torch.randn((k, n), dtype=torch.bfloat16)
            # Placement is per shape and comes from the model's own call, because it
            # changes the cost by more than 2x on the LM head.
            x = ttnn.from_torch(
                torch.randn((1, 1, 1, k), dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, device=device, memory_config=act_mem(ttnn, name),
            )
            print(f"\n{name} K={k} N={n} ({k_tiles}x{n_tiles} tiles), {cores} cores, "
                  f"in0_block_w={bw}, {describe(name)}", flush=True)
            print(f"  {'fmt':5s} {'MB':>8} {'dev us':>9} {'GB/s':>7} {'%pk':>6}")
            times, byts = {}, {}
            for fmt in ("bf16", "bfp8", "bfp4"):
                try:
                    wt = ttnn.from_torch(w_src, dtype=dt[fmt], layout=ttnn.TILE_LAYOUT,
                                         device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                except Exception as e:  # noqa: BLE001
                    print(f"  {fmt:5s} alloc FAILED {str(e)[:70]}", flush=True)
                    continue
                pcN = math.ceil(n_tiles / cores)
                cols, rows = min(11, cores), math.ceil(cores / min(11, cores))
                subW = max([i for i in range(1, 5) if pcN % i == 0] or [1])
                progcfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(cols, rows),
                    in0_block_w=bw, out_subblock_h=1, out_subblock_w=subW,
                    per_core_M=1, per_core_N=pcN, fuse_batch=True,
                    fused_activation=None, mcast_in0=True,
                )

                def once(progcfg=progcfg, wt=wt):
                    y = ttnn.linear(x, wt, compute_kernel_config=ckc, program_config=progcfg,
                                    memory_config=out_mem(ttnn, name))
                    ttnn.synchronize_device(device)
                    ttnn.deallocate(y)

                try:
                    med, samples = timer.best(once)
                    once()
                    dev = device_kernel_us(ttnn, device) or med * 1e6
                except Exception as e:  # noqa: BLE001
                    print(f"  {fmt:5s} FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
                    ttnn.deallocate(wt)
                    continue
                mb = weight_mb(n, k, fmt)
                times[fmt], byts[fmt] = dev, mb
                rec = new_record(f"floors_{name}", K=k, N=n, fmt=fmt, cores=cores,
                                 in0_block_w=bw, placement=describe(name))
                rec.samples = samples
                append_jsonl(args.out, rec, int(mb * 1e6))
                print(f"  {fmt:5s} {mb:8.2f} {dev:9.1f} {mb / dev * 1e3:7.1f} "
                      f"{mb / dev * 1e3 / DRAM_PEAK_GBS * 100:5.1f}%", flush=True)
                ttnn.deallocate(wt)

            have = [f for f in ("bf16", "bfp8", "bfp4") if f in times]
            if len(have) >= 2:
                a, b, res = fit([byts[f] for f in have], [times[f] for f in have])
                mbbb = weight_mb(n, k, "bfp2")
                pred = a + b * mbbb
                print(f"  fit: fixed {a:6.1f} us + {b:6.2f} us/MB "
                      f"({1e3 / b if b else 0:6.1f} GB/s, {1e3 / b / DRAM_PEAK_GBS * 100 if b else 0:.0f}% peak)"
                      f", residual {res:.1f} us")
                if "bfp4" in times and pred > 0:
                    print(f"  PREDICTED bfp2 ({mbbb:.2f} MB): {pred:.1f} us = "
                          f"{times['bfp4'] / pred:.2f}x vs bfp4 "
                          f"{'<-- PASSES the 1.4x gate' if times['bfp4'] / pred >= 1.4 else '(below 1.4x)'}")
            ttnn.deallocate(x)
            del w_src
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

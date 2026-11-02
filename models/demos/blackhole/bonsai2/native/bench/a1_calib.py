# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Calibrate the op benchmark against the model, on device kernel duration.

The planner's steer (11:33) is that the A1 fits may be measuring harness cost
rather than math: the op benchmark reads 28-49 us above the device kernel time of
the same matmuls inside the model. Two suspects, and this separates them:

  1. **The call is not the model's call.** The model passes no ``program_config``,
     so ttnn auto-selects a 1D mcast config sized to the core count it wants, and
     passes ``packer_l1_acc=1`` with LoFi. The A0/A1 sweeps passed no compute
     config and let the *DRAM-sharded* decode config happen via memory placement.
     Those are different programs.
  2. **The timing method.** A host 5-call median includes dispatch. The model's
     numbers come from the device profiler's device kernel duration.

So every row here is timed **both** ways -- host median and device kernel
duration from the profiler -- and run under **both** configs. If host and device
agree and the configs differ, the fix is the config. If host and device disagree,
the fix is the timing method. The model's own numbers are printed alongside so
the calibration is anchored to them rather than to my re-measurement of them:

    ffn_down    17408x5120  bfp8   88 cores   347.7 us
    ffn_gate    5120x17408  bfp4  110 cores   170.0 us
    gdn in-proj 5120x16480  bfp8   44 cores   232.5 us
    attn q+gate 5120x12288  bfp8   99 cores   167.6 us
    gdn out      6144x5120  bfp8   33 cores    88.4 us
    attn k/v     5120x1024  bfp8   33 cores    73.4 us
    attn o_proj  6144x5120  bfp8   88 cores   123.3 us
    lm_head     5120x248320 bfp8  DRAM -> DRAM   -        3718.0 us (measured, not in CSV)

Usage (on a chip)::

    python -m models.demos.blackhole.bonsai2.native.bench.a1_calib
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .placement import act_mem
from .placement import out_mem as out_mem_fn
from .record import DRAM_PEAK_GBS
from .timing import Timer

#: (name, K, N, model's measured device kernel us). K is the reduction dim.
#: The model's numbers are from the planner's TP1+norm decode profile at 14:49.
SHAPES = [
    ("ffn_down", 17408, 5120, 347.7),
    ("ffn_gate", 5120, 17408, 170.0),
    ("gdn_in_proj", 5120, 16480, 232.5),
    ("attn_q_gate", 5120, 12288, 167.6),
    ("gdn_out", 6144, 5120, 88.4),
    ("attn_kv", 5120, 1024, 73.4),
    ("attn_o", 6144, 5120, 123.3),
    # The profile CSV has no single lm_head row to quote; 3718 us is my measured
    # device time for it with the activation in DRAM (logs/a1cal2.log), which is
    # where the model puts it.
    ("lm_head", 5120, 248320, 3718.0),
]

#: Which dtype the model uses for each shape (mlp gate/up are bfp4, rest bfp8).
MODEL_DTYPE = {"ffn_gate": "bfp4", "ffn_down": "bfp8"}

#: Placement lives in one place (bench/placement.py) so the calibration and the
#: floors sweep cannot drift apart on it.


def device_kernel_us(ttnn, device) -> float | None:
    """Device kernel duration for the program since the last readback.

    Reads ``DEVICE KERNEL DURATION [ns]`` specifically. The analyses dict also
    holds per-RISC durations (BRISC, NCRISC, TRISC0-2) and FW durations, so a
    naive "sum everything with KERNEL and DURATION in the name" inflates the
    result by ~5x -- that mistake is why an earlier version of this bench
    reported 10.9 ms for a 73 us matmul.
    """
    try:
        ttnn.ReadDeviceProfiler(device)
        perf = ttnn.get_latest_programs_perf_data()
    except Exception:  # noqa: BLE001
        return None
    if not perf:
        return None
    key = "DEVICE KERNEL DURATION [ns]"
    for _chip, programs in perf.items():
        for program in programs:
            res = program.program_analyses_results.get(key)
            if res is not None:
                return int(res.duration) / 1e3
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="")
    ap.add_argument("--mode", default="model,auto-sharded",
                    help="model = replay the model's call; auto-sharded = the A0 config")
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--profile", action="store_true",
                    help="also read device kernel duration (needs the profiler env flags)")
    args = ap.parse_args()

    if args.profile:
        for k, v in (("TT_METAL_DEVICE_PROFILER", "1"),
                     ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
                     ("TT_METAL_PROFILER_MID_RUN_DUMP", "1")):
            os.environ.setdefault(k, v)

    import ttnn

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    timer = Timer(warmup=3, calls=args.calls, repeat=args.repeat)
    dt = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
    # The model's decode config (mlp.py): LoFi, fp32 dest acc, packer_l1_acc.
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    shapes = SHAPES
    if args.shapes:
        want = set(args.shapes.split(","))
        shapes = [s for s in shapes if s[0] in want]

    print(f"Peak {DRAM_PEAK_GBS} GB/s.  host = 5-call median; device = profiler kernel duration.")
    print(f"{'shape':12s} {'K':>6} {'N':>7} {'fmt':5s} {'MB':>7} "
          f"{'host us':>9} {'dev us':>8} {'GB/s':>7} {'%pk':>5} {'model us':>9} {'delta':>8}")
    try:
        for name, k, n, model_us in shapes:
            fmt = MODEL_DTYPE.get(name, "bfp8")
            w_src = torch.randn((k, n), dtype=torch.bfloat16)
            nb = n * k + n * k // 16 if fmt == "bfp8" else n * k // 2 + n * k // 16
            x_mem = act_mem(ttnn, name)
            for mode in args.mode.split(","):
                wt = ttnn.from_torch(
                    w_src, dtype=dt[fmt], layout=ttnn.TILE_LAYOUT,
                    device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                x = ttnn.from_torch(
                    torch.randn((1, 1, 1, k), dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=device, memory_config=x_mem,
                )
                out_mem = out_mem_fn(ttnn, name)

                if mode == "model":
                    # Exactly the model's call: no program config, its compute config,
                    # and the activation/output placement the model uses for this shape
                    # (bench/placement.py -- L1->L1 for most projections, DRAM in or out
                    # for attn_o and lm_head).
                    def once():
                        y = ttnn.linear(x, wt, compute_kernel_config=ckc, memory_config=out_mem)
                        ttnn.synchronize_device(device)
                        ttnn.deallocate(y)
                else:
                    # The earlier sweeps' call, kept as a control: no compute config,
                    # DRAM output. Placement of the input still follows the model so the
                    # two arms differ only in the call, not in where the data lives.
                    def once():
                        y = ttnn.linear(x, wt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                        ttnn.synchronize_device(device)
                        ttnn.deallocate(y)

                try:
                    med, samples = timer.best(once)
                except Exception as e:  # noqa: BLE001
                    print(f"{name:12s} {mode:14s} FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
                    ttnn.deallocate(wt)
                    ttnn.deallocate(x)
                    continue
                dev_us = float("nan")
                if args.profile:
                    once()
                    d = device_kernel_us(ttnn, device)
                    if d:
                        dev_us = d
                best_us = dev_us if dev_us == dev_us else med * 1e6
                gbs = nb / (best_us * 1e-6) / 1e9
                delta = f"{best_us - model_us:+8.1f}" if model_us else f"{'-':>8s}"
                print(f"{name:12s} {k:6d} {n:7d} {fmt:5s} {nb / 1e6:7.2f} "
                      f"{med * 1e6:9.1f} {dev_us:8.1f} {gbs:7.1f} "
                      f"{gbs / DRAM_PEAK_GBS * 100:4.1f}% "
                      f"{(f'{model_us:.1f}' if model_us else '-'):>9s} {delta}"
                      f"  [{mode}]", flush=True)
                ttnn.deallocate(wt)
                ttnn.deallocate(x)
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

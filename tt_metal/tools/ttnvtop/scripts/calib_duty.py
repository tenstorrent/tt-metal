#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# CALIBRATION: does the monitor's FPU/SFPU reading mean anything?
#
# Everything measured so far answers "did it crash" and "did it stay fresh". Neither says
# the NUMBER is true. Under a CCL workload F and S read ~0, which is plausible -- fabric
# collectives are NOC and DRAM movement, not FPU math -- but plausible is not measured.
#
# The trick is to avoid needing a peak-FLOP constant, which is where this kind of
# validation usually goes wrong (a wrong peak makes a correct counter look broken and vice
# versa). Instead sweep a KNOWN DUTY CYCLE: run matmuls back to back for a fraction d of
# each phase and idle the rest, with a device sync so the idle is real. Then
#
#     monitor_reading(d)  should be  LINEAR in d, through the origin.
#
# Linearity and a zero intercept are the whole test, and both are peak-free. The slope is
# the fraction of matmul wall time the FPU is actually issuing (< 1: a matmul also moves
# data), so it calibrates the reading without needing to know that fraction in advance.
#
# Emits phase boundaries as JSON so a separate SHM probe can be aligned to them; the two
# must stay separate processes because the monitor must never be in-process with the
# workload it measures.

import argparse
import json
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--size", type=int, default=2048, help="square matmul dim")
    ap.add_argument("--phase-s", type=float, default=40.0, help="seconds per duty phase")
    ap.add_argument("--duties", type=str, default="0,25,50,75,100")
    ap.add_argument("--out", type=str, default="/tmp/calib_phases.json")
    ap.add_argument("--sfpu", action="store_true", help="also drive SFPU (exp) per iteration")
    args = ap.parse_args()

    import torch
    import ttnn

    duties = [float(x) for x in args.duties.split(",")]
    phases = []

    device = ttnn.open_mesh_device(ttnn.MeshShape(args.rows, args.cols))
    try:
        n = args.size
        a = ttnn.from_torch(
            torch.randn(n, n, dtype=torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        b = ttnn.from_torch(
            torch.randn(n, n, dtype=torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

        # Warm up: first launch pays compile + program cache.
        for _ in range(8):
            c = ttnn.matmul(a, b)
            if args.sfpu:
                c = ttnn.exp(c)
            ttnn.deallocate(c)
        ttnn.synchronize_device(device)

        # One calibrated batch: how long does a sync'd burst of matmuls take? Needed so a
        # duty can be hit by choosing an idle time, rather than by guessing.
        BATCH = 4
        t0 = time.monotonic()
        for _ in range(BATCH):
            c = ttnn.matmul(a, b)
            if args.sfpu:
                c = ttnn.exp(c)
            ttnn.deallocate(c)
        ttnn.synchronize_device(device)
        batch_s = (time.monotonic() - t0) / BATCH
        print(f"# per-iteration busy time: {batch_s*1e3:.3f} ms", file=sys.stderr, flush=True)

        for duty in duties:
            t_start = time.monotonic()
            busy_accum = 0.0
            iters = 0
            deadline = t_start + args.phase_s
            if duty <= 0.0:
                # True idle phase: the floor. Whatever the monitor reports here is its
                # noise floor, and the intercept the linear fit must pass through.
                time.sleep(args.phase_s)
            else:
                # idle per iteration chosen so busy/(busy+idle) == duty
                idle_s = batch_s * (100.0 - duty) / duty
                while time.monotonic() < deadline:
                    tb = time.monotonic()
                    c = ttnn.matmul(a, b)
                    if args.sfpu:
                        c = ttnn.exp(c)
                    ttnn.deallocate(c)
                    ttnn.synchronize_device(device)
                    busy_accum += time.monotonic() - tb
                    iters += 1
                    if idle_s > 0:
                        time.sleep(idle_s)
            t_end = time.monotonic()
            span = t_end - t_start
            actual = 100.0 * busy_accum / span if span > 0 else 0.0
            phases.append(
                {
                    "duty_target": duty,
                    "duty_actual_host": actual,
                    "t_start": t_start,
                    "t_end": t_end,
                    "iters": iters,
                }
            )
            print(
                f"# phase duty={duty:5.1f}% -> host-measured busy {actual:5.1f}% " f"({iters} iters over {span:.1f}s)",
                file=sys.stderr,
                flush=True,
            )
    finally:
        ttnn.close_mesh_device(device)

    with open(args.out, "w") as f:
        json.dump({"size": args.size, "sfpu": args.sfpu, "phases": phases}, f, indent=1)
    print(json.dumps({"ok": True, "phases": len(phases), "out": args.out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

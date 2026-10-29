# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Timed HPCG solve on the P150: 50 traced CG iterations, residual history, GFLOP/s and share of speed of light.

usage: python models/experimental/hpcg/demo/demo.py nx [iterations]
Prints one JSON object per line. The CG start (r = b - A x, |r|) and one CG iteration are each captured in a trace
and replayed; nothing is read back while they run. ms_per_iteration: iterations replayed back to back, best of 3.
gflops is the raw rate: HPCG's flop count over the timed iterations. cg_start_ms: the CG start alone.

Where a CPU reference residual is known (CPU_REFERENCE_RESIDUAL_50), it also prints hpcg_rating_gflops, the
formula of HPCG's ReportResults.cpp for a long run. One CG set is timed as CG.cpp times it: the CG start, then the
k iterations this solver needs to reach the reference residual, at least 50 (cg_set_s, best of 3). Credited work per set =
(k iterations + the CG start) * 50 / k flops; time per set = cg_set_s + (setup time) / 10. Setup here is building the
matrix, index tables and programs on the host (HPCG's GenerateProblem + OptimizeProblem).
"""

import json
import os
import sys
import time

# TT-Metal's inspector appends a log record for every program it creates; on ext4 such an append can wait ~30 ms for
# a journal commit (seen as a rare 30+ ms first launch). A timed run turns it off unless the caller sets it.
os.environ.setdefault("TT_METAL_INSPECTOR", "0")

import numpy as np  # noqa: E402
import ttnn  # noqa: E402

from models.experimental.hpcg.reference import speed_of_light  # noqa: E402
from models.experimental.hpcg.tt.hpcg import HPCG  # noqa: E402
from models.experimental.hpcg.tt.layout import from_device  # noqa: E402

# HPCG 3.1 reference code (fp64, 24 threads, Ryzen 9 7900X): residual after 50 iterations
CPU_REFERENCE_RESIDUAL_50 = {104: 4.99963e-08, 128: 3.945e-07, 192: 5.315e-05}


def emit(**kw):
    print(json.dumps(kw), flush=True)


def main(nx, iters):
    device = ttnn.open_device(device_id=0, trace_region_size=512 << 20)
    try:
        t0 = time.time()
        h = HPCG(device, nx)
        ttnn.synchronize_device(device)  # setup includes the device's matrix generation
        setup_s = time.time() - t0
        emit(
            nx=nx,
            setup_s=round(setup_s, 3),
            rows=[L.n for L in h.levels],
            nnz=[L.nnz for L in h.levels],
            bands=[L.bands for L in h.levels],
        )
        h.reset()
        h.start()  # compiles every program
        h.iteration()
        ttnn.synchronize_device(device)
        tid_start = ttnn.begin_trace_capture(device, cq_id=0)
        normr0 = h.start()
        ttnn.end_trace_capture(device, tid_start, cq_id=0)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        normr = h.iteration()
        ttnn.end_trace_capture(device, tid, cq_id=0)

        def solve(n, time_start, record=False):
            """x = 0 (untimed, as in HPCG), then the CG start and n iterations. time_start: the timer covers the CG
            start too, as HPCG's CG.cpp does; otherwise it covers the iterations only."""
            h.reset()
            if not time_start:
                ttnn.execute_trace(device, tid_start, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            ts = time.perf_counter()
            if time_start:
                ttnn.execute_trace(device, tid_start, cq_id=0, blocking=False)
            hist = []
            nr0 = float(ttnn.to_torch(normr0).reshape(-1)[0]) if record else None
            for _ in range(n):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
                if record:
                    hist.append(float(ttnn.to_torch(normr).reshape(-1)[0]) / nr0)
            ttnn.synchronize_device(device)
            return time.perf_counter() - ts, hist

        ms = min(solve(iters, False)[0] for _ in range(3)) / iters * 1e3
        start_ms = min(solve(0, True)[0] for _ in range(3)) * 1e3
        sol = {o["op"]: o for o in speed_of_light.ops(nx)}
        sol_ms = sum(o["per_iter"] * o["sol_us"] for o in sol.values()) / 1e3
        flops = speed_of_light.hpcg_flops(nx)
        emit(
            ms_per_iteration=round(ms, 3),
            gflops=round(flops / (ms * 1e-3) / 1e9, 2),
            sol_ms=round(sol_ms, 3),
            pct_of_sol=round(100 * sol_ms / ms, 1),
            cg_start_ms=round(start_ms, 3),
        )
        _, hist = solve(max(iters, 60), False, record=True)
        out = dict(residual_history=[float(f"{r:.6e}") for r in hist])
        if nx in CPU_REFERENCE_RESIDUAL_50:
            hit = [k + 1 for k, r in enumerate(hist) if r <= CPU_REFERENCE_RESIDUAL_50[nx]]
            out["iterations_to_cpu_reference_residual"] = hit[0] if hit else None
            if hit:
                # one CG set as HPCG times it: the CG start and k iterations, at least 50 (main.cpp's optNiters starts
                # at refMaxIters = 50, so a solver that converges sooner still runs 50)
                k = max(hit[0], 50)
                per_set = min(solve(k, True)[0] for _ in range(3))
                # ReportResults.cpp: credited flops = (k iterations + the CG start) * 50 / k
                credit = (k * flops + speed_of_light.hpcg_start_flops(nx)) * 50 / k
                out["cg_set_s"] = round(per_set, 4)
                out["rating_without_setup_gflops"] = round(credit / per_set / 1e9, 2)
                out["hpcg_rating_gflops"] = round(credit / (per_set + setup_s / 10) / 1e9, 2)
        L0 = h.levels[0]
        out["max_abs_x_minus_1"] = float(np.max(np.abs(from_device(h.x)[L0.real] - 1)))
        emit(**out)
        ttnn.release_trace(device, tid)
        ttnn.release_trace(device, tid_start)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 50)

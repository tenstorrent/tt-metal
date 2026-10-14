# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Timed HPCG solve on the P150: 50 traced CG iterations, residual history, GFLOP/s and share of speed of light.

usage: python models/experimental/hpcg/demo/demo.py nx [iterations]
Prints one JSON object per line. Timing: the CG start (r = b - A x) runs untraced, then one traced iteration is
replayed back to back; best of 3. Host setup (matrix and index tables) is excluded, as in HPCG's raw rating.
"""

import json
import sys
import time

import numpy as np
import ttnn

from models.experimental.hpcg.reference import speed_of_light
from models.experimental.hpcg.tt.hpcg import HPCG
from models.experimental.hpcg.tt.layout import from_device

# HPCG 3.1 reference code (fp64, 24 threads, Ryzen 9 7900X): residual after 50 iterations
CPU_REFERENCE_RESIDUAL_50 = {104: 4.99963e-08, 128: 3.945e-07, 192: 5.315e-05}


def emit(**kw):
    print(json.dumps(kw), flush=True)


def main(nx, iters):
    device = ttnn.open_device(device_id=0, trace_region_size=512 << 20)
    try:
        t0 = time.time()
        h = HPCG(device, nx)
        emit(
            nx=nx,
            setup_s=round(time.time() - t0, 1),
            rows=[L.n for L in h.levels],
            nnz=[L.nnz for L in h.levels],
            bands=[L.bands for L in h.levels],
        )
        h.setup_cg()  # compiles every program
        h.iteration()
        ttnn.synchronize_device(device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        normr = h.iteration()
        ttnn.end_trace_capture(device, tid, cq_id=0)

        def solve(n, record=False):
            h.setup_cg()
            ttnn.synchronize_device(device)
            hist = []
            ts = time.perf_counter()
            for _ in range(n):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
                if record:
                    hist.append(float(ttnn.to_torch(normr).reshape(-1)[0]) / h.normr0)
            ttnn.synchronize_device(device)
            return time.perf_counter() - ts, hist

        best = min(solve(iters)[0] for _ in range(3))
        ms = best / iters * 1e3
        sol = {o["op"]: o for o in speed_of_light.ops(nx)}
        sol_ms = sum(o["per_iter"] * o["sol_us"] for o in sol.values()) / 1e3
        flops = speed_of_light.hpcg_flops(nx)
        emit(
            ms_per_iteration=round(ms, 3),
            gflops=round(flops / (ms * 1e-3) / 1e9, 2),
            sol_ms=round(sol_ms, 3),
            pct_of_sol=round(100 * sol_ms / ms, 1),
        )
        _, hist = solve(max(iters, 60), record=True)
        out = dict(residual_history=[float(f"{r:.6e}") for r in hist])
        if nx in CPU_REFERENCE_RESIDUAL_50:
            hit = [k + 1 for k, r in enumerate(hist) if r <= CPU_REFERENCE_RESIDUAL_50[nx]]
            out["iterations_to_cpu_reference_residual"] = hit[0] if hit else None
        L0 = h.levels[0]
        out["max_abs_x_minus_1"] = float(np.max(np.abs(from_device(h.x)[L0.real] - 1)))
        emit(**out)
        ttnn.release_trace(device, tid)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 50)

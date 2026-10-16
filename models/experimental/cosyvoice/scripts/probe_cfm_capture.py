# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""How much of the flow stage is trace capture rather than trace replay?

The flow stage is not linear in solver depth (`probe_flow_steps.py`), so part of each
call is a fixed cost. Without the trace cache, `solve_euler` captures and releases the
estimator trace on every call. This measures the capture/replay split, which decides
whether keeping the trace per mel length pays (it does: `COSYVOICE_CFM_TRACE_CACHE`,
PERF.md Part II §2.2).

    python3 models/experimental/cosyvoice/scripts/probe_cfm_capture.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

MEL, FRAMES, STEPS = 80, 282, 10


def main() -> int:
    from models.experimental.cosyvoice.tt.flow.cfm import TtConditionalCFM, cosine_t_span, euler_steps
    from models.experimental.cosyvoice.tt.weights import WeightBag, default_weights_path

    path = default_weights_path().replace("hift_", "flow_")
    device = ttnn.open_device(device_id=0, l1_small_size=131072, trace_region_size=402653184)
    try:
        bag = WeightBag.load(path)
        meta = bag.meta
        cfm = TtConditionalCFM(
            device, bag.sub("decoder"), inference_cfg_rate=meta.get("inference_cfg_rate", 0.7), n_timesteps=STEPS
        )
        torch.manual_seed(0)
        fixed = [
            torch.randn(1, FRAMES, MEL) * 0.1,
            torch.randn(1, FRAMES, MEL) * 0.1,
            torch.randn(1, 1, MEL) * 0.1,
            torch.randn(1, FRAMES, MEL) * 0.1,
        ]

        def dev(v):
            return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Warm every kernel first, so what follows measures capture and replay rather
        # than JIT compilation.
        ttnn.deallocate(cfm.solve_euler(*(dev(v) for v in fixed)))
        ttnn.synchronize_device(device)

        schedule = euler_steps(cosine_t_span(STEPS, cfm.t_scheduler))
        results = {}
        for rep in range(3):
            x, mu, spks, cond = (dev(v) for v in fixed)
            mu2 = cfm._cfg_pair(mu, zero_second_row=True)
            spks2 = cfm._cfg_pair(spks, zero_second_row=True)
            cond2 = cfm._cfg_pair(cond, zero_second_row=True)
            ttnn.synchronize_device(device)

            t0 = time.perf_counter()
            cfm._capture(x, mu2, spks2, cond2, schedule[0][0], schedule[0][1])
            ttnn.synchronize_device(device)
            t_cap = time.perf_counter() - t0

            ts = [dev(torch.full((2, 1, 1), t, dtype=torch.float32)) for t, _ in schedule]
            dts = [dev(torch.full((1, 1, 1), dt, dtype=torch.float32)) for _, dt in schedule]
            ttnn.synchronize_device(device)

            t0 = time.perf_counter()
            for t_dev, dt_dev in zip(ts, dts):
                ttnn.copy(t_dev, cfm._t_buf)
                ttnn.copy(dt_dev, cfm._dt_buf)
                ttnn.execute_trace(device, cfm._trace_id, cq_id=0, blocking=True)
                ttnn.copy(cfm._next_x, cfm._x_buf)
            ttnn.synchronize_device(device)
            t_rep = time.perf_counter() - t0

            t0 = time.perf_counter()
            cfm._release()
            ttnn.synchronize_device(device)
            t_rel = time.perf_counter() - t0

            for t in (*ts, *dts, mu2, spks2, cond2):
                ttnn.deallocate(t)
            if rep == 0 or t_cap + t_rep + t_rel < sum(results.values()):
                results = {"capture": t_cap, "replay (10 steps)": t_rep, "release": t_rel}

        total = sum(results.values())
        print(f"\n  flow solver, {FRAMES} frames, {STEPS} Euler steps -- best of 3")
        print(f"  {'phase':<22}{'s':>9}{'share':>9}")
        print("  " + "-" * 40)
        for k, v in results.items():
            print(f"  {k:<22}{v:>9.4f}{v / total * 100:>8.1f}%")
        print(f"  {'total':<22}{total:>9.4f}")
        amortised = results["replay (10 steps)"]
        print(f"\n  Capture + release is paid once per utterance and is inside the measured")
        print(f"  flow time. Caching the trace per mel-length bucket would leave {amortised:.4f} s")
        print(f"  for every utterance after the first in that bucket -- {total / amortised:.2f}x on this stage.")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

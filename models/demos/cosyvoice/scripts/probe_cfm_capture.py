# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""How much of the flow stage is trace *capture* rather than trace *replay*?

`probe_flow_steps.py` swept the solver depth and found the stage is not linear in it:
fitting the seven points gives **T(n) ~ 0.350 s + 35.8 ms per Euler step**. Ten steps is
`0.708 s`, of which only `0.358 s` is the ODE. The other half is a fixed cost that no
change to `n_timesteps` can touch, which is why cutting the solver from 10 steps to 5
buys 1.43x rather than 2x.

Reading `solve_euler` says where it goes: it calls `self._capture(...)` at line 294 and
`self._release()` at line 342, so **the estimator trace is captured and thrown away on
every call**. One utterance, one capture. That is correct and it is also the single
largest remaining op-level cost in the model -- larger than anything the op-level work so
far addressed in the flow.

The question this settles is whether that cost is *inherent* or *amortisable*:

  - Inherent, if capture must happen per utterance because shapes change with mel length.
  - Amortisable, if a trace can be cached per length bucket the way `cache_width()`
    already buckets the LLM's KV cache to a multiple of 128.

The second would take the flow stage to roughly its replay cost for every utterance after
the first in a bucket. This measures the split so the size of that prize is known before
anyone builds it.

    python3 models/demos/cosyvoice/scripts/probe_cfm_capture.py
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
    from models.demos.cosyvoice.tt.flow.cfm import TtConditionalCFM, cosine_t_span, euler_steps
    from models.demos.cosyvoice.tt.weights import WeightBag, default_weights_path

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
        # than JIT compilation -- the trap `07_cookbook.md` records twice.
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

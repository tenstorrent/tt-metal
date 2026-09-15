# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""What does the flow solver's depth cost, and what does cutting it buy?

The flow decoder is `0.603 s` of a `1.470 s` total — 41 %, and the largest single
obstacle to `RTF < 0.2`, which allows `0.654 s` for everything. Its cost is **linear in
`n_timesteps`**: ten forward-Euler steps, each one full estimator evaluation, and nothing
else in the stage scales with it.

So the arithmetic is inviting, and that is exactly the shape of claim this project has
learned to distrust. Two questions have to be answered together:

  1. Is the speed-up actually linear, or does a fixed per-call cost dominate at low step
     counts? An earlier length sweep found the flow's cost-per-tile *falls* with length,
     the signature of a
     large fixed cost — if that cost is per-*stage* rather than per-*step*, halving the
     steps buys much less than half.
  2. What does the output lose? The checkpoint ships `n_timesteps = 10` and every
     accuracy figure in PERF.md is measured there. Fewer steps is a **coarser ODE
     solve**, not a numerical shortcut — the error is in the trajectory, not the
     arithmetic, so PCC against the 10-step result is the honest metric.

The issue asks for this experiment directly: *"optimize iterative refinement process /
consider approximations for faster inference"*. It does not say the approximation is
acceptable, which is what this measures.

    COSYVOICE_FLOW_STEPS is the knob; this sweeps it directly.
    python3 models/demos/cosyvoice/scripts/probe_flow_steps.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

STEPS = (10, 8, 6, 5, 4, 3, 2, 1)
MEL, FRAMES = 80, 282  # the benchmark utterance
REPS = 3


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main() -> int:
    from models.demos.cosyvoice.tt.flow.cfm import TtConditionalCFM, cosine_t_span
    from models.demos.cosyvoice.tt.weights import WeightBag, default_weights_path

    path = default_weights_path().replace("hift_", "flow_")
    device = ttnn.open_device(device_id=0, l1_small_size=131072, trace_region_size=402653184)
    try:
        bag = WeightBag.load(path)
        meta = bag.meta
        cfm = TtConditionalCFM(
            device,
            bag.sub("decoder"),
            inference_cfg_rate=meta.get("inference_cfg_rate", 0.7),
            n_timesteps=10,
        )

        # One fixed set of inputs, drawn once. The *same* initial noise must feed every
        # step count, or the comparison measures the draw rather than the solver — the
        # discipline `capture-rng-draws` exists for.
        torch.manual_seed(0)
        fixed = [
            torch.randn(1, FRAMES, MEL) * 0.1,
            torch.randn(1, FRAMES, MEL) * 0.1,
            torch.randn(1, 1, MEL) * 0.1,
            torch.randn(1, FRAMES, MEL) * 0.1,
        ]

        def dev(v):
            return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        def run(n: int):
            span = cosine_t_span(n, cfm.t_scheduler)
            args = tuple(dev(v) for v in fixed)
            out = cfm.solve_euler(*args, t_span=span)
            got = ttnn.to_torch(out).float()
            ttnn.deallocate(out)
            return got

        def timed(n: int) -> float:
            span = cosine_t_span(n, cfm.t_scheduler)
            ttnn.deallocate(cfm.solve_euler(*(dev(v) for v in fixed), t_span=span))  # warm + capture
            ttnn.synchronize_device(device)
            best = None
            for _ in range(REPS):
                t0 = time.perf_counter()
                out = cfm.solve_euler(*(dev(v) for v in fixed), t_span=span)
                ttnn.synchronize_device(device)
                best = min(best or 1e9, time.perf_counter() - t0)
                ttnn.deallocate(out)
            return best

        ref = run(10)
        print(f"\n  {'steps':>6}{'s':>9}{'ms/step':>10}{'vs 10 steps':>14}{'PCC vs 10':>14}")
        print("  " + "-" * 55)
        base = None
        for n in STEPS:
            t = timed(n)
            base = base or t
            p = 1.0 if n == 10 else pcc(run(n), ref)
            print(f"  {n:>6}{t:>9.4f}{t * 1e3 / n:>10.1f}{base / t:>13.2f}x{p:>14.8f}")

        print("\n  ms/step flat        -> cost is genuinely per-step; the trade is linear.")
        print("  ms/step rising      -> a fixed per-stage cost dominates and fewer steps buys less.")
        print("  PCC is against the **shipped 10-step result**, which is the reference this")
        print("  port is gated on -- not against a torch golden, because at fewer steps the")
        print("  solver is answering a different question and the golden no longer applies.")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

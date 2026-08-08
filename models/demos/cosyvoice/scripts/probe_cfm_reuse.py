# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Does keeping the CFM trace across utterances stay correct, and what does it save?

`probe_cfm_capture.py` measured the split: of a `0.675 s` flow solve, **`0.314 s` is
trace capture** and `0.357 s` is the ten Euler replays. Capture was being paid on every
call and thrown away.

Reusing it needs the conditioning refilled *in place*, because the trace holds
`_packed_const`'s address. The failure mode if that is wrong is the nastiest kind
available here: the replay reads the **previous utterance's** conditioning and produces
fluent audio in the wrong voice. No exception, no shape mismatch, and a per-module PCC
against a single golden would not see it either.

So the test is deliberately built to catch exactly that: **three consecutive solves with
different conditioning**, cached against uncached, compared solve by solve. A stale
`_packed_const` shows up as solve 2 and 3 disagreeing while solve 1 matches.

    python3 models/demos/cosyvoice/scripts/probe_cfm_reuse.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

MEL, FRAMES, N = 80, 282, 3


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main() -> int:
    from models.demos.cosyvoice.tt.flow.cfm import TtConditionalCFM
    from models.demos.cosyvoice.tt.weights import WeightBag, default_weights_path

    path = default_weights_path().replace("hift_", "flow_")
    device = ttnn.open_device(device_id=0, l1_small_size=131072, trace_region_size=402653184)
    try:
        bag = WeightBag.load(path)
        meta = bag.meta

        # Three *different* conditionings at one mel length -- the case the cache is
        # supposed to serve, and the case a stale buffer would silently break.
        torch.manual_seed(0)
        cases = [
            (
                torch.randn(1, FRAMES, MEL) * 0.1,
                torch.randn(1, FRAMES, MEL) * 0.1,
                torch.randn(1, 1, MEL) * 0.1,
                torch.randn(1, FRAMES, MEL) * 0.1,
            )
            for _ in range(N)
        ]

        def dev(v):
            return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        def sweep(cache: bool):
            os.environ["COSYVOICE_CFM_TRACE_CACHE"] = "1" if cache else "0"
            cfm = TtConditionalCFM(
                device, bag.sub("decoder"), inference_cfg_rate=meta.get("inference_cfg_rate", 0.7), n_timesteps=10
            )
            outs, times = [], []
            for case in cases:
                args = tuple(dev(v) for v in case)
                ttnn.synchronize_device(device)
                t0 = time.perf_counter()
                out = cfm.solve_euler(*args)
                ttnn.synchronize_device(device)
                times.append(time.perf_counter() - t0)
                outs.append(ttnn.to_torch(out).float())
                ttnn.deallocate(out)
            cfm._release()
            return outs, times

        off_out, off_t = sweep(False)
        on_out, on_t = sweep(True)

        print(f"\n  {N} consecutive solves, {FRAMES} mel frames, different conditioning each time")
        print(f"  {'solve':>7}{'no cache (s)':>15}{'cached (s)':>13}{'speedup':>10}{'PCC':>14}")
        print("  " + "-" * 60)
        for i in range(N):
            print(
                f"  {i:>7}{off_t[i]:>15.4f}{on_t[i]:>13.4f}"
                f"{off_t[i] / on_t[i]:>9.2f}x{pcc(on_out[i], off_out[i]):>14.10f}"
            )
        steady_off = sum(off_t[1:]) / (N - 1)
        steady_on = sum(on_t[1:]) / (N - 1)
        print(
            f"\n  steady state (solves 1..{N-1}): {steady_off:.4f} -> {steady_on:.4f} s  ({steady_off/steady_on:.2f}x)"
        )
        print("  PCC 1.0 on every solve  -> the conditioning refill is correct.")
        print("  PCC 1.0 on solve 0 only -> stale `_packed_const`; the cache is wrong.")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

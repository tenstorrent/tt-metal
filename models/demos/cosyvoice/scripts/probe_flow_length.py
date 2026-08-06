# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Is the flow decoder's cost non-monotonic in mel length, the way the AR step is?

F45 found that a decode step costs what the *parity* of its key-axis tile count says,
not what its size says: 10/12/14/16 tiles cost 6.32/6.73/7.09/7.99 ms while 11/13/15
cost 7.33/7.92/8.54. Padding a tensor up to an even tile count was worth about a
millisecond on a 6.7 ms step.

The flow is now the second-largest stage -- 0.602 s of a 1.675 s total, 36 % -- and its
activations are `[B, T, C]` with `T` on a tiled axis. This utterance is **282 mel
frames, which pads to 9 tiles: odd**. If the same effect applies, padding the solver to
320 frames (10 tiles) would be a pure win despite moving 13 % more data.

Whether it applies is a real question rather than a formality, because the two stages
sit in different regimes. The AR step is dispatch-bound on tiny tensors, where an extra
tile costs almost nothing to move and a great deal to schedule badly. The flow moves
`[2, 282, 320]`-scale activations through 10 Euler steps, and if it is compute-bound
then 13 % more data is simply 13 % more time and the idea is dead.

So: sweep the length, and read the shape of the curve rather than any single point.
Saw-toothed means build it; smooth and rising means stop.

    python models/demos/cosyvoice/scripts/probe_flow_length.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

# Around the real 282: tile counts 8, 9 (the current one), 10, 11, 12.
LENGTHS = (256, 282, 288, 320, 352, 384)
MEL = 80
REPS = 3


def main() -> int:
    from models.demos.cosyvoice.tt.flow.cfm import TtConditionalCFM
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
            n_timesteps=meta.get("n_timesteps", 10),
        )
        torch.manual_seed(0)

        def dev(v):
            return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        print(f"\n  {'mel frames':>11}{'tiles':>7}{'s / 10 steps':>14}{'ms per tile':>13}")
        print("  " + "-" * 45)
        for t in LENGTHS:
            args = lambda: (  # noqa: E731 -- solve_euler consumes x, so rebuild per call
                dev(torch.randn(1, t, MEL) * 0.1),
                dev(torch.randn(1, t, MEL) * 0.1),
                dev(torch.randn(1, 1, MEL) * 0.1),
                dev(torch.randn(1, t, MEL) * 0.1),
            )
            ttnn.deallocate(cfm.solve_euler(*args()))  # warm + capture
            ttnn.synchronize_device(device)
            best = None
            for _ in range(REPS):
                t0 = time.perf_counter()
                out = cfm.solve_euler(*args())
                ttnn.synchronize_device(device)
                # Best of N, not mean: this is a like-for-like comparison across
                # lengths, and a slow run is host noise rather than a property of
                # the length being measured.
                best = min(best or 1e9, time.perf_counter() - t0)
                ttnn.deallocate(out)
            tiles = (t + 31) // 32
            print(f"  {t:>11}{tiles:>7}{best:>14.4f}{best * 1e3 / tiles:>13.1f}")

        print("\n  Saw-toothed (odd tile counts dearer) -> pad the solver to an even tile count.")
        print("  Smooth and rising                    -> the flow is compute-bound; F45 does not apply.")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

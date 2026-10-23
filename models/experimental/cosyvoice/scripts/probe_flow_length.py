# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Is the flow decoder's cost non-monotonic in mel length, the way the AR step's is?

A decode step's cost tracks the parity of its key-axis tile count rather than its size
(`TracedDecodeStepInPlace`), so padding to an even tile count can pay. The flow's
activations are `[B, T, C]` with `T` on a tiled axis, and the captured utterance's 282
mel frames pad to 9 tiles, an odd count; if the same effect applies, padding the solver
to 320 frames (10 tiles) pays despite moving 13 % more data. The flow runs much larger
tensors than the AR step, so if it is compute-limited, 13 % more data is 13 % more
time.

This sweeps the length. Read the shape of the curve rather than any single point:
saw-toothed means pad, smooth and rising means do not.

    python models/experimental/cosyvoice/scripts/probe_flow_length.py
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
    from models.experimental.cosyvoice.tt.flow.cfm import TtConditionalCFM
    from models.experimental.cosyvoice.tt.weights import WeightBag, default_weights_path

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
        print("  Smooth and rising                    -> the flow is compute-bound; tile parity does not apply.")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""`ttnn.snake_beta` already exists. What is it worth here?

`ttnn.snake` was proposed as the contribution portfolio's opener -- "the strongest
opener", "no tt-llk change", propose it first. `tt/hifigan/snake.py` says the same
thing in its docstring: *"TTNN has no native `snake`, so it is composed from primitives
here."*

Both are out of date. `ttnn.snake_beta` landed in **PR #43614 on 2026-05-26**, ten weeks
before that proposal was written, as a ternary SFPU op at
`eltwise/ternary/ternary_nanobind.cpp:337`. It computes `x + sin^2(alpha*x)/beta`, so
`snake_beta(x, alpha, alpha)` is exactly the Snake this vocoder wants, and its
broadcasting contract -- alpha and beta non-1 only on the **last** dimension -- is
exactly the channels-last `[B, T, C]` layout `conv.py` already uses.

So it is not a contribution to propose. The question that replaces it is whether swapping the
composed five-op form for the native one is worth anything, at the two shapes HiFT
actually runs: after the first upsample (256 channels, 2256 frames) and after the second
(128 channels, 18048 frames). 48 activations per vocoder call.

    python3 models/demos/cosyvoice/scripts/probe_snake_native.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

# (frames, channels) -- HiFT after ups[0] and after ups[1], plus the pre-upsample body.
SHAPES = ((282, 512), (2256, 256), (18048, 128))
CALLS = 48  # Snake activations per vocoder call: 8 ResBlocks x 6
REPS = 5


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main() -> int:
    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=402653184)
    if not hasattr(ttnn, "snake_beta"):
        print("  ttnn.snake_beta is absent from this build -- nothing to measure.")
        ttnn.close_device(device)
        return 1

    print(f"\n  {'shape':>18}{'composed (ms)':>15}{'snake_beta (ms)':>17}{'speedup':>9}{'PCC':>14}")
    print("  " + "-" * 74)

    for t, c in SHAPES:
        torch.manual_seed(0)
        x_t = torch.randn(1, t, c) * 0.5
        alpha_t = torch.rand(1, 1, c) * 0.9 + 0.1  # HiFT alphas are positive, O(1)

        def dev(v):
            return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        x, alpha = dev(x_t), dev(alpha_t)
        inv_alpha = dev(1.0 / (alpha_t + 1e-9))  # folded on host, as tt/hifigan/snake.py does

        # What ships: multiply -> sin -> square -> multiply -> add, four intermediates.
        def composed():
            out = x
            for _ in range(CALLS):
                s = ttnn.multiply(out, alpha)
                s2 = ttnn.sin(s)
                ttnn.deallocate(s)
                s3 = ttnn.square(s2)
                ttnn.deallocate(s2)
                s4 = ttnn.multiply(s3, inv_alpha)
                ttnn.deallocate(s3)
                nxt = ttnn.add(out, s4)
                ttnn.deallocate(s4)
                if out is not x:
                    ttnn.deallocate(out)
                out = nxt
            return out

        # `beta = alpha` turns SnakeBeta into Snake. The host-folded reciprocal is not
        # needed: the op divides by beta itself.
        def native():
            out = x
            for _ in range(CALLS):
                nxt = ttnn.snake_beta(out, alpha, alpha)
                if out is not x:
                    ttnn.deallocate(out)
                out = nxt
            return out

        a_out, b_out = composed(), native()
        p = pcc(ttnn.to_torch(a_out).float(), ttnn.to_torch(b_out).float())

        timings = {}
        for label, body, held0 in (("composed", composed, a_out), ("native", native, b_out)):
            ttnn.deallocate(held0)
            ttnn.deallocate(body())  # warm before capture
            ttnn.synchronize_device(device)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            held = body()
            ttnn.end_trace_capture(device, tid, cq_id=0)
            best = None
            for _ in range(REPS):
                t0 = time.perf_counter()
                ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
                best = min(best or 1e9, time.perf_counter() - t0)
            timings[label] = best * 1e3
            ttnn.release_trace(device, tid)
            ttnn.deallocate(held)

        shape = f"[1,{t},{c}]"
        print(
            f"  {shape:>18}{timings['composed']:>15.3f}{timings['native']:>17.3f}"
            f"{timings['composed'] / timings['native']:>8.2f}x{p:>14.8f}"
        )
        for tns in (x, alpha, inv_alpha):
            ttnn.deallocate(tns)

    print("\n  Times are for all 48 activations a vocoder call issues, at one shape.")
    print("  The real vocoder spreads them across the three, so the saving is a blend.")
    ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

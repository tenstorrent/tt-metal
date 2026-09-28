#!/usr/bin/env python3
"""Does measured power depend on per-call duration? A duty-cycle probe for tt-ember.

Telemetry samples every ~103 us while op calls take 32-470 us, so an individual call is never
resolved. That is fine only if the device is busy for essentially the whole window -- if each
call is followed by host dispatch dead time, the averaged power is diluted and understates the
op's true active power.

This runs the same op at increasing sizes so per-call duration sweeps from well under the
sample interval to far above it. If the duty cycle is already high, dynamic power is flat
across sizes and the original per-op numbers stand. If power climbs with call duration, the
short-call measurements were diluted and by how much.

Emits tt-ember's summary rows, so it is driven by unmodified auto.py.
"""
import argparse
import os
import sys
import time
from datetime import datetime

if not os.environ.get("TT_METAL_HOME"):
    sys.stderr.write("TT_METAL_HOME is not set.\n")
    raise SystemExit(2)

import torch  # noqa: E402
import ttnn  # noqa: E402


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


def build_cases(device):
    def T(*shape):
        return ttnn.from_torch(
            torch.randn(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    cases = []
    # Elementwise: cost scales with element count, so size sweeps call duration directly.
    for s, h in [(1024, 2048), (4096, 4096), (8192, 8192), (16384, 8192)]:
        x = T(1, 1, s, h)
        cases.append((f"add_{s}x{h}", lambda x=x: ttnn.add(x, x), 0))
    # Matmul: square-ish, cost ~ s*h*h.
    for s, h in [(1024, 2048), (2048, 4096), (4096, 4096), (4096, 8192)]:
        a, b = T(1, 1, s, h), T(1, 1, h, h)
        cases.append((f"matmul_{s}x{h}", lambda a=a, b=b: ttnn.matmul(a, b), 2 * s * h * h))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-seconds", type=float, default=4.0)
    ap.add_argument("--pause-seconds", type=float, default=5.0)
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    print("=== Duty-cycle probe: power vs per-call duration ===", flush=True)
    device = ttnn.open_device(device_id=args.device_id)
    rows = []
    try:
        for idx, (name, fn, flops) in enumerate(build_cases(device), start=1):
            if idx > 1:
                time.sleep(args.pause_seconds)

            fn()
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            for _ in range(3):
                fn()
            ttnn.synchronize_device(device)
            per_iter = (time.perf_counter() - t0) / 3
            iters = max(1, int(args.target_seconds / per_iter))

            print(f"# OP {idx} {name} iters={iters} flops_per_iter={flops}", flush=True)
            ttnn.synchronize_device(device)
            start = now_str()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            ttnn.synchronize_device(device)
            elapsed = time.perf_counter() - t0
            end = now_str()

            tflops = (flops * iters / elapsed / 1e12) if flops else 0.0
            rows.append((idx, iters, elapsed, tflops, elapsed / iters * 1000.0, start, end))
            print(f"# DONE {name}: {elapsed/iters*1e6:.1f} us/call over {iters} iters", flush=True)
    finally:
        ttnn.close_device(device)

    print()
    print(f"{'Grid':>8} {'Cores':>7} {'Iters':>10} {'Time [s]':>12} {'TFLOPS':>10} "
          f"{'Per iter [ms]':>16} {'Start Time':>27} {'End Time':>27}")
    for idx, iters, elapsed, tflops, per_iter_ms, start, end in rows:
        print(f"{f'{idx}x1':>8} {idx:>7} {iters:>10} {elapsed:>12.6f} {tflops:>10.2f} "
              f"{per_iter_ms:>16.6f} {start:>27} {end:>27}")
    print()
    print("Test Passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

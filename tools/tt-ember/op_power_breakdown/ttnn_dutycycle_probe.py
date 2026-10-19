#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Does measured power depend on per-call duration? A duty-cycle probe for tt-ember.

Telemetry samples every ~103 us while op calls take 32-470 us, so an individual call is never
resolved. That is fine only if the device is busy for essentially the whole window -- if each
call is followed by host dispatch dead time, the averaged power is diluted and understates the
op's true active power.

This runs the same op at increasing sizes so per-call duration sweeps from well under the
sample interval to far above it. If the duty cycle is already high, dynamic power is flat
across sizes and the original per-op numbers stand. If power climbs with call duration, the
short-call measurements were diluted and by how much.

Emits tt-ember's summary rows (with ttnn_ops_workload.py's helpers), so it is driven by
unmodified auto.py, like the workload. It runs on device 0.
"""
import argparse
import sys
import time

# Checks TT_METAL_HOME and imports ttnn and torch.
import ttnn_ops_workload as wl  # noqa: E402

torch = wl.torch
ttnn = wl.ttnn


def build_cases(device):
    def T(*shape):
        return ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

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
    args = ap.parse_args()

    print("=== Duty-cycle probe: power vs per-call duration ===", flush=True)
    device = ttnn.open_device(device_id=0)
    rows = []
    try:
        for idx, (name, fn, flops) in enumerate(build_cases(device), start=1):
            if idx > 1:
                time.sleep(args.pause_seconds)

            iters = max(1, int(args.target_seconds / wl.time_call(fn, device)))
            print(f"# OP {idx} {name} iters={iters} flops_per_iter={flops}", flush=True)
            elapsed, start, end = wl.run_interval(fn, device, iters)
            tflops = (flops * iters / elapsed / 1e12) if flops else 0.0
            rows.append((idx, iters, elapsed, tflops, start, end))
            print(f"# DONE {name}: {elapsed/iters*1e6:.1f} us/call over {iters} iters", flush=True)
    finally:
        ttnn.close_device(device)

    wl.emit_summary_rows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

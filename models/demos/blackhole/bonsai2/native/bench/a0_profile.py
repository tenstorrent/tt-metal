# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where does the compressed op's time go: the DRAM reader or the compute loop?

A0 established that cost is flat in bytes and flat in format, and set mainly by
the number of output columns. Two structures could produce that, and they need
opposite fixes:

  reader    the NCRISC streaming loop's per-subblock semaphore/transaction
            bookkeeping: proportional to (columns x subblocks), independent of
            tile size
  compute   the TRISC loop's per-column tile_regs / cb_wait_front handoff:
            proportional to columns only, also independent of tile size

Varying K separates them without touching the kernel. Both loops scale with
subblocks in K, but the compute path also does real MVMUL work per K tile while
the reader only moves bytes -- so:

  * time flat in K at fixed columns  => pure handoff, no work: compute-side
    synchronisation is not even proportional to K, so it is per-column setup
  * time linear in K but with a large
    constant per column              => the constant is the per-column handoff
                                        and the slope is DRAM/LLK throughput

Also reports subblocks-per-column, so the per-subblock cost can be read off
directly. All runs use one core per bank (cpb=1), which A0 showed is best.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a0_profile
"""

from __future__ import annotations

import argparse
import os

from .a0_limits import build_M
from .record import DRAM_PEAK_GBS
from .timing import Timer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096, help="columns, fixed so only K moves")
    ap.add_argument("--sbk", type=int, default=16)
    ap.add_argument("--calls", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    import ttnn

    N = args.n
    sbk = args.sbk
    timer = Timer(warmup=2, calls=args.calls, repeat=args.repeat)
    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    try:
        print(f"N={N} columns fixed, bfp2, cpb=1, sbk={sbk}.  Peak {DRAM_PEAK_GBS} GB/s\n")
        rows = []
        # K must be a multiple of 32*sbk so the subblock divides evenly.
        for k in (512, 1024, 2048, 5120):
            if (k // 32) % sbk:
                print(f"  K={k}: skipped, Kt={k // 32} not divisible by sbk={sbk}")
                continue
            nsub = (k // 32) // sbk
            fn, nbytes, n_pad, pcnt = build_M(ttnn, device, N, k, 1, "bfp2", 1, sbk)
            med, _ = timer.best(fn)
            rows.append((k, nsub, med, nbytes))
            print(
                f"  K={k:5d}  subblocks/col={nsub:3d}  {med * 1e6:9.1f} us  "
                f"{nbytes / 1e6:7.2f} MB  {nbytes / med / 1e9:7.1f} GB/s  "
                f"{med * 1e6 / n_pad:7.4f} us/col"
            )

        if len(rows) >= 2:
            # Linear fit of time against subblocks-per-column, extrapolated to zero
            # subblocks: the intercept is the per-column cost with no work at all.
            xs = [r[1] for r in rows]
            ys = [r[2] * 1e6 / N for r in rows]  # us per column
            nn = len(xs)
            sx, sy = sum(xs), sum(ys)
            sxx = sum(x * x for x in xs)
            sxy = sum(x * y for x, y in zip(xs, ys))
            denom = nn * sxx - sx * sx
            slope = (nn * sxy - sx * sy) / denom if denom else 0.0
            intercept = (sy - slope * sx) / nn
            n_pad = rows[0][3]
            print(
                f"\nfit (us/col) = {intercept:.4f} + {slope:.4f} * subblocks_per_col\n"
                f"  intercept {intercept:.3f} us/col is pure per-column handoff\n"
                f"  slope     {slope:.4f} us per subblock per column"
            )
            # Bytes per subblock per column = sbk tiles x 320 B.
            bpsub = sbk * 320
            print(
                f"  slope implies {bpsub / (slope * 1e-6) / 1e9:.2f} GB/s of streaming throughput\n"
                f"  at the deepest K measured, the handoff share is "
                f"{intercept / (intercept + slope * rows[-1][1]) * 100:.1f}% of the time"
            )
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

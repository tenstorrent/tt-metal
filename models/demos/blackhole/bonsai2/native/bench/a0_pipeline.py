# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-side scaling of the compressed op: can the kernel reach the roofline?

``a0_deviceprofile`` established that 90% of the A0 wall clock was host-side
metadata rebuilding, and that the device kernel itself runs at about
1.6 GB/s per core (13 GB/s aggregate on 8 cores, 2.6% of peak) with the reader
and the compute loop balanced.

Balanced reader and compute usually means neither is the limiter but the
pipeline between them is too shallow: the reader only keeps ``num_buffers``
subblocks in flight, so in-flight bytes are ``3 * sbk * 320``.  At 16 tiles that
is 15 KB; sustaining 64 GB/s per core at a ~1 us round trip needs closer to
64 KB.  If that is the cause, a larger ``subblock_k`` shows up as a higher
device-side bandwidth, and A1 should use the largest subblock that still
divides Kt.

This measures device-side kernel time (not wall clock) across ``subblock_k`` and
``cores_per_bank``, so the pipeline-depth question is answered directly.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a0_pipeline
"""

from __future__ import annotations

import argparse
import os

from .a0_deviceprofile import _analyse
from .a0_limits import build_M
from .record import DRAM_PEAK_GBS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--k", type=int, default=5120)
    ap.add_argument("--sbk", default="16,40,80,160")
    ap.add_argument("--cpb", default="1")
    args = ap.parse_args()

    for k, v in (
        ("TT_METAL_DEVICE_PROFILER", "1"),
        ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
        ("TT_METAL_PROFILER_MID_RUN_DUMP", "1"),
    ):
        os.environ[k] = v

    import ttnn

    N, K = args.n, args.k
    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    try:
        print(f"K={K} N={N}, bfp2, device-side kernel time.  Peak {DRAM_PEAK_GBS} GB/s\n")
        print(f"{'sbk':>5} {'cpb':>4} {'in-flight':>10} {'reader us':>10} {'compute us':>11} {'MB':>7} {'GB/s':>7} {'%peak':>6}")
        for cpb in [int(x) for x in args.cpb.split(",")]:
            for sbk in [int(x) for x in args.sbk.split(",")]:
                if (K // 32) % sbk:
                    print(f"{sbk:5d} {cpb:4d}  skipped (Kt={K // 32} not divisible)")
                    continue
                try:
                    fn, nbytes, n_pad, pcnt = build_M(ttnn, device, N, K, 1, "bfp2", cpb, sbk)
                    fn()
                    agg = _analyse(ttnn, device)
                except Exception as e:  # noqa: BLE001
                    print(f"{sbk:5d} {cpb:4d}  FAILED {type(e).__name__}: {e}")
                    continue

                reader = agg.get("DEVICE NCRISC KERNEL DURATION [ns]", 0) / 1e3
                compute = agg.get("DEVICE TRISC0 KERNEL DURATION [ns]", 0) / 1e3
                # Use the slower of reader/compute as the kernel's real duration; they overlap.
                dur = max(reader, compute)
                gbs = nbytes / (dur * 1e-6) / 1e9 if dur else 0.0
                inflight = 3 * sbk * 320
                print(
                    f"{sbk:5d} {cpb:4d} {inflight / 1024:9.1f}K {reader:10.1f} {compute:11.1f} "
                    f"{nbytes / 1e6:7.2f} {gbs:7.2f} {gbs / DRAM_PEAK_GBS * 100:5.1f}%"
                )
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

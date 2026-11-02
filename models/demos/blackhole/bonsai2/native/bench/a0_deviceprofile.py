# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-side profiling of the compressed op: where the cycles actually go.

The K sweep in ``a0_profile`` fits ``us/col = 0.31 + 0.095 * subblocks`` and
implies about 54 GB/s per core of streaming throughput -- but a per-call host
cost proportional to columns x subblocks would produce exactly the same shape,
because the op rebuilds its per-core metadata tensors in Python on every call.
So the fit alone cannot say whether the limit is the device or my harness.

This uses tt-metal's device profiler, which times the kernels on the cores, to
get the real per-RISC breakdown: the NCRISC reader, the compute kernels, and
the gap between them. With that, the reader and the compute loop can be
compared directly instead of inferred.

Requires the full CI flag set (the same combination ``enable_profiler()`` sets):
``TT_METAL_DEVICE_PROFILER=1``, ``TT_METAL_PROFILER_CPP_POST_PROCESS=1``,
``TT_METAL_PROFILER_MID_RUN_DUMP=1``. Without MID_RUN_DUMP the data is never
consumed and the readback blocks.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a0_deviceprofile
"""

from __future__ import annotations

import argparse
import os

from .a0_limits import build_M
from .record import DRAM_PEAK_GBS
from .timing import Timer


def _analyse(ttnn, device) -> dict[str, int]:
    """Per-kernel-name total duration (ns) from the latest profiler readback."""
    ttnn.ReadDeviceProfiler(device)
    perf = ttnn.get_latest_programs_perf_data()
    agg: dict[str, int] = {}
    cores = 0
    if not perf:
        return agg
    for _chip, programs in perf.items():
        for program in programs:
            cores = max(cores, int(getattr(program, "core_count", 0) or 0))
            for name, result in program.program_analyses_results.items():
                agg[name] = agg.get(name, 0) + int(result.duration)
    if cores:
        agg["CORE COUNT"] = cores
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--k", type=int, default=5120)
    ap.add_argument("--sbk", type=int, default=16)
    ap.add_argument("--calls", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    flags = {
        "TT_METAL_DEVICE_PROFILER": "1",
        "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
        "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
    }
    for k, v in flags.items():
        if os.environ.get(k) != v:
            print(f"warning: {k}={os.environ.get(k)!r}; the readback needs the full CI flag set (setting {v})")
            os.environ[k] = v

    import ttnn

    N, K, sbk = args.n, args.k, args.sbk
    timer = Timer(warmup=1, calls=args.calls, repeat=args.repeat)
    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    try:
        fn, nbytes, n_pad, pcnt = build_M(ttnn, device, N, K, 1, "bfp2", 1, sbk)
        print(f"K={K} N={N} sbk={sbk} bfp2 cpb=1, {nbytes / 1e6:.2f} MB, {n_pad // 8 // 32} tile-columns/core")

        med, _ = timer.best(fn)
        print(f"wall clock (incl. host + dispatch): {med * 1e6:.1f} us")

        # One profiled call, so the readback covers exactly one program.
        fn()
        agg = _analyse(ttnn, device)
        if not agg:
            print("no profiler data returned")
            return 1

        total = sum(v for k, v in agg.items() if k != "CORE COUNT")
        print(f"\ndevice-side kernel time (sum over analyses, {agg.get('CORE COUNT', '?')} cores):")
        for name in sorted(agg, key=lambda k: -agg[k] if k != "CORE COUNT" else 0):
            if name == "CORE COUNT":
                continue
            share = agg[name] / total * 100 if total else 0
            print(f"  {name:44s} {agg[name] / 1e3:10.1f} us  {share:5.1f}%")

        # What the reader would take at full bandwidth, for comparison.
        ideal_us = nbytes / (DRAM_PEAK_GBS * 1e9 / 8) * 1e6  # per core, 8 banks
        print(
            f"\n{agg.get('CORE COUNT', '?')} cores streaming {nbytes / 8 / 1e6:.2f} MB each at "
            f"{DRAM_PEAK_GBS / 8:.0f} GB/s/core would take {ideal_us:.1f} us"
        )
        print(f"measured device total {total / 1e3:.1f} us is {total / 1e3 / ideal_us:.1f}x that")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

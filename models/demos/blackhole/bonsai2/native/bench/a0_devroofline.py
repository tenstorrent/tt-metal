# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A0 roofline, final form: device-side GB/s for every Bonsai projection shape.

Supersedes the wall-clock version. ``a0_deviceprofile`` showed the compressed
op's per-call host cost (rebuilding per-core metadata in Python) is about 10x
its device time, so wall clock says almost nothing about streaming throughput.
This reports the profiler's own reader/compute durations instead, which is the
roofline the plan actually asks for, per shape.

For each shape the kernel is profiled as-is (no per-call rebuild to blame), and
the result is quoted as the slower of reader and compute, since they overlap:
that is the kernel's real duration.

Output: one JSON line per (shape, sbk) through ``bench.record``, so every number
carries the commit, chip, env and firmware, plus the full per-RISC breakdown.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a0_devroofline \
        --gguf ~/avan/models/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf
"""

from __future__ import annotations

import argparse
import os

from .a0_deviceprofile import _analyse
from .a0_limits import build_M
from .a0_roofline import SHAPES
from .record import DRAM_PEAK_GBS, append_jsonl, new_record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="", help="unused; shapes are synthetic trits here")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--sbk", default="160", help="tiles per subblock; 160 divides Kt=160")
    ap.add_argument("--cpb", type=int, default=1)
    ap.add_argument("--out", default=os.path.expanduser("~/avan/native2bit/agentA/logs/a0_devroofline.jsonl"))
    args = ap.parse_args()

    for k, v in (
        ("TT_METAL_DEVICE_PROFILER", "1"),
        ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
        ("TT_METAL_PROFILER_MID_RUN_DUMP", "1"),
    ):
        os.environ[k] = v

    import ttnn
    from .. import gguf_ternary as gt

    shapes = SHAPES
    if args.shapes:
        want = set(args.shapes.split(","))
        shapes = [s for s in shapes if s[0] in want]

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    rows = []
    try:
        print(f"device-side streaming bandwidth, bfp2, cpb={args.cpb}.  Peak {DRAM_PEAK_GBS} GB/s")
        print(f"{'shape':12s} {'K':>6} {'N':>7} {'MB':>8} {'reader':>9} {'compute':>9} {'GB/s':>8} {'%peak':>7}")
        for name, k, n in shapes:
            for sbk in [int(x) for x in args.sbk.split(",")]:
                if (k // 32) % sbk:
                    continue
                try:
                    fn, nbytes, n_pad, pcnt = build_M(ttnn, device, n, k, 1, "bfp2", args.cpb, sbk)
                    fn()
                    agg = _analyse(ttnn, device)
                except Exception as e:  # noqa: BLE001
                    print(f"{name:12s} K={k} N={n} sbk={sbk}: FAILED {type(e).__name__}: {e}")
                    continue
                reader = agg.get("DEVICE NCRISC KERNEL DURATION [ns]", 0) / 1e3
                tris0 = agg.get("DEVICE TRISC0 KERNEL DURATION [ns]", 0) / 1e3
                tris2 = agg.get("DEVICE TRISC2 KERNEL DURATION [ns]", 0) / 1e3
                dur = max(reader, tris0)
                gbs = nbytes / (dur * 1e-6) / 1e9 if dur else 0.0
                rec = new_record(
                    f"a0dev_{name}",
                    K=k,
                    N=n,
                    N_padded=n_pad,
                    tiles_per_core=pcnt,
                    subblock_k=sbk,
                    cores_per_bank=args.cpb,
                    method="device profiler, max(reader,compute)",
                    reader_us=reader,
                    compute_us=tris0,
                    pack_us=tris2,
                    gguf=os.path.basename(args.gguf),
                )
                rec.samples = [dur * 1e-6]
                print(
                    f"{name:12s} {k:6d} {n:7d} {nbytes / 1e6:8.2f} {reader:9.1f} {tris0:9.1f} "
                    f"{gbs:8.1f} {gbs / DRAM_PEAK_GBS * 100:6.1f}%"
                )
                append_jsonl(args.out, rec, nbytes)
                rows.append((name, k, n, nbytes, gbs))

        # Whole-token total: every projection once, at its measured bandwidth.
        total_bytes = sum(r[3] for r in rows if r[0] not in ("lm_head",))
        total_us = sum(r[3] / (r[4] * 1e9) * 1e6 for r in rows)
        print(
            f"\nsum over these {len(rows)} projections: {total_bytes / 1e9:.2f} GB in {total_us / 1e3:.3f} ms "
            f"=> {1e6 / total_us:.1f} tok/s if the engine were the only cost"
        )
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

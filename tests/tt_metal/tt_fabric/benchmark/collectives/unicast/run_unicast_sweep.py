# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse, csv, re, subprocess, sys
from pathlib import Path


def parse_sizes(s: str):
    return [int(x) for x in re.split(r"[,\s]+", s.strip()) if x]


def parse_core(s: str):
    x, y = s.split(",")
    return int(x), int(y)


def parse_targets(s: str):
    out = {}
    if not s:
        return out
    for pair in re.split(r"[,\s]+", s.strip()):
        if not pair:
            continue
        sz, val = pair.split(":")
        out[int(sz)] = float(val)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="build/test/tt_metal/tt_fabric/bench_unicast")
    ap.add_argument("--src", default="0:0")
    ap.add_argument("--dst", default="0:1")
    ap.add_argument("--page", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--send-core", default="0,0")
    ap.add_argument("--recv-core", default="0,0")
    ap.add_argument("--sizes", default="4096,32768,1048576")
    ap.add_argument("--out-dir", default="generated/profiler/.logs/unicast")
    ap.add_argument("--csv", default="")
    ap.add_argument("--min-p50-GB-s", type=float, default=None)  # simple single floor
    ap.add_argument("--p50-targets", type=str, default="")  # per-size: "4096:0.04,32768:0.28,1048576:2.45"
    ap.add_argument("--tolerance-pct", type=float, default=5.0)
    ap.add_argument(
        "--trace-iters",
        type=int,
        default=1,
        help="number of enqueues captured per trace (batch size for replay timing)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else (out_dir / "unicast_sweep.csv")

    sizes = parse_sizes(args.sizes)
    rx, ry = parse_core(args.recv_core)
    targets = parse_targets(args.p50_targets)

    for size in sizes:
        cmd = [
            args.bin,
            "--src-dev",
            args.src,
            "--dst-dev",
            args.dst,
            "--size",
            str(size),
            "--page",
            str(args.page),
            "--send-core",
            args.send_core,
            "--recv-core",
            f"{rx},{ry}",
            "--iters",
            str(args.iters),
            "--warmup",
            str(args.warmup),
            "--csv",
            str(csv_path),
            "--trace-iters",
            str(args.trace_iters),
        ]
        print(">>", " ".join(cmd))
        subprocess.run(cmd, check=True)

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("FAIL: CSV has no rows")

    if targets:
        failed = False
        for size in sizes:
            # last row for this size (in case CSV already had prior runs)
            row = next((r for r in reversed(rows) if int(r["sizeB"]) == size), None)
            if row is None:
                print(f"FAIL: no row for size {size}", file=sys.stderr)
                failed = True
                continue
            measured = float(row["p50_GB_s"])
            if size in targets:
                lo = targets[size] * (1.0 - args.tolerance_pct / 100.0)
                if measured < lo:
                    print(
                        f"FAIL size {size}: p50_GB_s={measured:.3f} < {lo:.3f} "
                        f"(target {targets[size]:.3f} -{args.tolerance_pct:.1f}%)",
                        file=sys.stderr,
                    )
                    failed = True
                else:
                    print(
                        f"PASS size {size}: p50_GB_s={measured:.3f} ≥ {lo:.3f} "
                        f"(target {targets[size]:.3f} -{args.tolerance_pct:.1f}%)"
                    )
            else:
                print(f"NOTE: no target for size {size}, measured p50={measured:.3f} GB/s")
        if failed:
            sys.exit(1)
    elif args.min_p50_GB_s is not None:
        last_p50 = float(rows[-1]["p50_GB_s"])
        if last_p50 < args.min_p50_GB_s:
            sys.exit(f"FAIL: p50_GB_s={last_p50} < min={args.min_p50_GB_s}")
        print(f"PASS: p50_GB_s={last_p50} ≥ min={args.min_p50_GB_s}")
    else:
        print("NOTE: no CI guard thresholds provided.")


if __name__ == "__main__":
    main()

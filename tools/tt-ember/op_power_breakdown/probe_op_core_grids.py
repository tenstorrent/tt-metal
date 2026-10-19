#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""How many Tensix cores does each op of the block actually occupy?

The power breakdown cannot answer this. tt-ember measures board power, and a low reading is
equally consistent with "all cores busy at low activity" and "few cores busy" -- which matter
very differently. Occupancy is a static property of the op and shape, so it is probed here in
a separate pass rather than during the power run, where the profiler's overhead would distort
the measurement.

Uses tt-metal's device profiler, which is compiled in by default (build_metal.sh has
--disable-profiler, not --enable-profiler). Each op is run a couple of times with
TT_METAL_DEVICE_PROFILER=1; ttnn.ReadDeviceProfiler() flushes per-core kernel records to
profile_log_device.csv (only with TT_METAL_PROFILER_MID_RUN_DUMP=1, which this sets), and the
rows added by each op are counted by their distinct (core_x, core_y).

Run it the same way as the workload -- TT_METAL_HOME set, tt-metal venv active. The op list
follows LONG_MATMUL_MODE as in the workload; the batches below are the published p100a run's,
so they go with the legacy list:

    TT_METAL_DEVICE_PROFILER=1 LONG_MATMUL_MODE=legacy python3 probe_op_core_grids.py \\
        --seq 1024 --hidden 2048 --ffn 8192 --heads 16 --batches 32,7,7,7,3,14,21,35,5,9,10,5
"""
import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path

if not os.environ.get("TT_METAL_HOME"):
    sys.stderr.write("TT_METAL_HOME is not set.\n")
    raise SystemExit(2)
if os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
    sys.stderr.write("Set TT_METAL_DEVICE_PROFILER=1 so the device profiler records anything.\n")
    raise SystemExit(2)
# Without it ReadDeviceProfiler() keeps the records until the device closes, and every op would
# read as zero cores.
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")

import ttnn  # noqa: E402

# Where tt-metal writes the CSV: $TT_METAL_PROFILER_DIR (relative to this process's cwd, as
# tt-metal resolves it), else <tt-metal root>/generated/profiler.
_PROFILER_DIR = os.environ.get("TT_METAL_PROFILER_DIR")
CSV_PATH = (
    (Path(_PROFILER_DIR).resolve() if _PROFILER_DIR else Path(os.environ["TT_METAL_HOME"]) / "generated" / "profiler")
    / ".logs"
    / "profile_log_device.csv"
)


def load_workload(path: Path):
    spec = importlib.util.spec_from_file_location("ttnn_ops_workload", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cores_in_rows(rows, hdr):
    ix, iy = hdr.index("core_x"), hdr.index("core_y")
    return {(r[ix].strip(), r[iy].strip()) for r in rows if len(r) > max(ix, iy)}


def read_csv_rows():
    if not CSV_PATH.exists():
        return [], []
    rows = list(csv.reader(open(CSV_PATH)))
    if len(rows) < 2:
        return [], []
    return [h.strip() for h in rows[1]], rows[2:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", type=Path, default=Path(__file__).with_name("ttnn_ops_workload.py"))
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--ffn", type=int, default=8192)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument(
        "--batches",
        default="",
        help="Comma-separated batch per op, in block order. Defaults to 1 for all. "
        "Use the batches the power run actually chose, from its # OP lines.",
    )
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    wl = load_workload(args.workload)
    legacy = wl.legacy_mode()

    if CSV_PATH.exists():
        CSV_PATH.unlink()

    device = ttnn.open_device(device_id=args.device_id)
    grid = device.compute_with_storage_grid_size()
    total_cores = grid.x * grid.y
    results = []
    try:
        factories = wl.make_factories(device, args.seq, args.hidden, args.ffn, args.heads, legacy)
        batches = [int(b) for b in args.batches.split(",")] if args.batches else []
        seen_before = 0

        for idx, (name, factory) in enumerate(factories, start=1):
            b = batches[idx - 1] if idx - 1 < len(batches) else 1
            while True:
                try:
                    fn, _, _ = factory(b)
                    break
                except Exception as e:
                    if b == 1 or not wl.is_out_of_memory(e):
                        raise
                    b = max(1, b // 2)

            for _ in range(args.reps):
                fn()
            ttnn.synchronize_device(device)
            ttnn.ReadDeviceProfiler(device)

            hdr, rows = read_csv_rows()
            new = rows[seen_before:]
            seen_before = len(rows)
            if not new:
                raise SystemExit(
                    f"{name}: no new profiler rows in {CSV_PATH}; check TT_METAL_PROFILER_DIR and that "
                    "TT_METAL_PROFILER_MID_RUN_DUMP is not 0"
                )
            cores = cores_in_rows(new, hdr) if hdr else set()
            results.append((name, b, len(cores)))
            print(f"{name:<16} batch={b:<3d} cores={len(cores):3d}", flush=True)
            del fn
    finally:
        ttnn.close_device(device)

    print()
    print(f"{'op':<16} {'batch':>6} {'cores':>6} {'% of ' + str(total_cores):>9}")
    print("-" * 42)
    for name, b, n in results:
        print(f"{name:<16} {b:>6} {n:>6} {100*n/total_cores:>8.0f}%")


if __name__ == "__main__":
    main()

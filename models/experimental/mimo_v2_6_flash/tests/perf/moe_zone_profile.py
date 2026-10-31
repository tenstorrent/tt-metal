# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where the time of the deduplicated MoE streams goes: the MOE_MARK points of tt/kernels/moe_*_uniq_* (reader,
writer, compute of GateUpStream and DownStream) at the speculative-verify shapes of kagent_moe_verify_bench.py, from
the device profiler.

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=<dir> python moe_zone_profile.py

Runs CALLS eager forward_decode calls per shape on Galaxy row 0, reads the profiler, then prints per program and
point the time since the program's first kernel start (us; median over the 8 bank cores and the calls after the
first, and the latest core), and each program's kernel span."""

import csv
import os
import statistics
from collections import defaultdict

import torch

import ttnn
from models.experimental.mimo_v2_6_flash.tests.perf import moe_decode_common as common

CALLS = 4
USERS = 8
SHAPES = ("one_request", "two_requests")
POINTS = {
    10: "GU reader start",
    11: "GU reader staged",
    12: "GU reader routed (local users)",
    13: "GU reader took remote users",
    14: "GU reader stream end",
    20: "GU writer start",
    21: "GU writer x read",
    22: "GU writer staged",
    23: "GU writer routed (own users)",
    24: "GU writer got local users",
    25: "GU writer count pushed",
    26: "GU writer weight tiles pushed",
    27: "GU writer first h written",
    28: "GU writer end (fused: every h landed)",
    29: "fused writer end",
    30: "GU compute start",
    31: "GU compute got count",
    32: "GU compute first expert done",
    33: "GU compute end (fused: gate|up done)",
    34: "fused compute h ready (down starts)",
    35: "fused compute end",
    40: "DN reader start",
    41: "DN reader ids",
    42: "DN reader stream end",
    50: "DN writer start",
    51: "DN writer ids",
    52: "DN writer h read",
    53: "DN writer end",
    60: "DN compute start",
    61: "DN compute got count and h",
    62: "DN compute end",
}


def read_log(path):
    """Rows of profile_log_device.csv: (chip, core, risc, cycles, data, run id, zone, type); and the clock in MHz."""
    with open(path) as f:
        mhz = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = []
        for r in csv.DictReader(f, skipinitialspace=True):
            rows.append(
                (
                    int(r["PCIe slot"]),
                    (int(r["core_x"]), int(r["core_y"])),
                    r["RISC processor type"],
                    int(r["time[cycles since reset]"]),
                    int(r["data"] or 0),
                    int(r["run host ID"] or 0),
                    r["zone name"],
                    r["type"],
                )
            )
    return rows, mhz


def report(rows, mhz):
    chip = min(r[0] for r in rows)
    runs = defaultdict(list)
    for r in rows:
        if r[0] == chip:
            runs[r[5]].append(r)
    programs = defaultdict(list)  # "GU" / "DN" -> per run: {point: [us per core]}, span us
    for rid in sorted(runs):
        rr = runs[rid]
        marks = [r for r in rr if r[6] == "MOE" and r[7] == "TS_DATA"]
        if not marks:
            continue
        starts = [r[3] for r in rr if r[6].endswith("-KERNEL") and r[7] == "ZONE_START"]
        ends = [r[3] for r in rr if r[6].endswith("-KERNEL") and r[7] == "ZONE_END"]
        t0 = min(starts)
        points = defaultdict(list)
        for r in marks:
            # compute points: the math TRISC's
            if r[4] // 10 in (3, 6) and r[2] != "TRISC_1":
                continue
            points[r[4]].append((r[3] - t0) / mhz)
        kind = "GU" if min(points) < 40 else "DN"
        programs[kind].append((points, (max(ends) - t0) / mhz))
    return programs


def main():
    out_dir = os.environ["TT_METAL_PROFILER_DIR"]
    torch.manual_seed(0)
    cfg = common.config()
    inputs = common.golden_moe_inputs(cfg, common.checkpoint(cfg))
    steps = {name: idx for name, idx in common.verify_steps(cfg, 4)[4:8] if name in SHAPES}
    mesh, row = common.open_row()
    try:
        moe = common.build_moe(row, cfg, "model_bfp4experts", ttnn.bfloat4_b, users=USERS)
        for name in SHAPES:
            x, mask = common.step_inputs(row, inputs, steps[name])
            for _ in range(CALLS):
                moe.forward_decode(x, mask).deallocate(True)
                ttnn.synchronize_device(row)
            x.deallocate(True)
            mask.deallocate(True)
        ttnn.ReadDeviceProfiler(row)
    finally:
        common.close(mesh, row)

    rows, mhz = read_log(os.path.join(out_dir, ".logs", "profile_log_device.csv"))
    programs = report(rows, mhz)
    for kind, runs in programs.items():
        for s, name in enumerate(SHAPES):
            mine = runs[s * CALLS + 1 : (s + 1) * CALLS]  # the first call of a shape warms
            spans = [span for _, span in mine]
            print(f"\n== {kind} {name}: kernel span {statistics.median(spans):.1f} us")
            for p in sorted({p for pts, _ in mine for p in pts}):
                med = statistics.median(statistics.median(pts[p]) for pts, _ in mine if p in pts)
                late = statistics.median(max(pts[p]) for pts, _ in mine if p in pts)
                print(f"  {p:3d} {POINTS.get(p, '?'):34s} median core {med:7.1f}  latest core {late:7.1f}")


if __name__ == "__main__":
    main()

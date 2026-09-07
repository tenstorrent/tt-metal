# Perf experiment `stream_regime` -- where does the time go, per stage?
#
# Runs ONE case under ONE variant with RMS_STAGE_ZONES=1 and aggregates the
# per-stage device zones, so the candidate's remaining headroom is read off the
# stages rather than guessed at.  Occupancy zones (reader_read_x, writer_write)
# contain their own CB stalls -- they are WAIT + WORK, not payload.
import collections
import csv
import os
import statistics
import sys
from pathlib import Path

os.environ["RMS_STAGE_ZONES"] = "1"

REPO = Path("/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal")
HERE = Path(os.environ.get("RMS_EXP_DIR") or Path(__file__).resolve().parent)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tests/ttnn/unit_tests/operations/rms_norm_ttnn/probes"))

import bench_stream  # noqa: E402  (sets up CASES + VARIANTS)
import bench_r3  # noqa: E402
import ttnn  # noqa: E402

CSVP = Path("generated/profiler/.logs/profile_log_device.csv")
FREQ = 1.35


def parse_zones():
    rows = []
    with CSVP.open() as fh:
        fh.readline()
        rdr = csv.reader(fh)
        header = [h.strip() for h in next(rdr)]
        idx = {h: i for i, h in enumerate(header)}
        for r in rdr:
            if len(r) >= len(header):
                rows.append(r)
    ci = {
        k: idx[k]
        for k in (
            "core_x",
            "core_y",
            "RISC processor type",
            "time[cycles since reset]",
            "zone name",
            "type",
            "run host ID",
        )
    }
    target = sorted({int(r[ci["run host ID"]]) for r in rows})[-1]
    open_stack = collections.defaultdict(list)
    per_core = collections.defaultdict(lambda: collections.defaultdict(float))
    counts = collections.defaultdict(int)
    for r in rows:
        if int(r[ci["run host ID"]]) != target:
            continue
        key = (r[ci["core_x"]], r[ci["core_y"]], r[ci["RISC processor type"]])
        zone = r[ci["zone name"]].strip()
        t = int(r[ci["time[cycles since reset]"]])
        if r[ci["type"]].strip() == "ZONE_START":
            open_stack[(key, zone)].append(t)
        elif r[ci["type"]].strip() == "ZONE_END" and open_stack[(key, zone)]:
            dt = t - open_stack[(key, zone)].pop()
            per_core[(zone, r[ci["RISC processor type"]])][key] += dt
            counts[(zone, r[ci["RISC processor type"]])] += 1
    return per_core, counts, target


def main():
    case = os.environ.get("RMS_CASES", "T15_7168_gbr32")
    variant = os.environ.get("RMS_VARIANTS", "rowres")
    bench_stream.VARIANTS[variant]()
    if CSVP.exists():
        CSVP.unlink()
    device = ttnn.open_device(device_id=0)
    try:
        ns, p, r = bench_r3.measure(device, case)
        print(f"RESULT.zone case={case} variant={variant} ns={ns:.0f} pcc={p:.6f} relrms={r:.3e}", flush=True)
    finally:
        ttnn.close_device(device)
    per_core, counts, target = parse_zones()
    print(f"RESULT ---- zones ({case} / {variant}, run {target}) ----")
    print(f"RESULT {'zone':28s} {'risc':8s} {'occ/core':>10s} {'max/core':>10s} {'n/core':>8s}")
    out = []
    for k, cores in per_core.items():
        vals = list(cores.values())
        out.append((statistics.mean(vals), k[0], k[1], max(vals), counts[k] / len(vals)))
    for mean, z, ri, mx, n in sorted(out, reverse=True):
        print(f"RESULT {z:28s} {ri:8s} {mean/FREQ:10.1f} {mx/FREQ:10.1f} {n:8.2f}")


main()

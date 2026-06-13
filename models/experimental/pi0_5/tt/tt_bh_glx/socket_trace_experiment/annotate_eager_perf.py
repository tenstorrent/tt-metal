# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Annotate the EAGER (no-trace) signposted ops_perf_results CSV by phase.

run_socket_traced.py EAGER=1 emits tracy signposts around each phase:
  (ops before PHASE_warmup)  -> init_one_time   (pipeline build, weights, upstream)
  PHASE_warmup               -> warm-up         (full pipeline, programs compiling)
  PHASE_prefix_vision        -> iter:vision      (profiled iter starts here)
  PHASE_prefix_build         -> iter:build_prefix
  PHASE_prefix_prefill       -> iter:prefill
  PHASE_kv_migration         -> iter:kv_migration
  PHASE_denoise              -> iter:denoise
  PHASE_end                  -> teardown

Signpost rows appear in the CSV with OP TYPE=="signpost" and the name in OP CODE,
interleaved with ops by HOST START TS. We walk the timeline, label every op with the
most-recent signpost's phase, and also tag each op's STAGE by device id.

Outputs (next to this script's --out dir):
  socket_e2e_ops_annotated.csv   — per-op, trimmed key columns + PHASE + STAGE
  socket_e2e_phase_summary.csv   — per-phase rollup (#ops, device-kernel ms, host ms, ...)

Usage:
  python annotate_eager_perf.py <ops_perf_results.csv> <out_dir>
"""

import csv
import os
import sys
from collections import defaultdict

PHASE_LABEL = {
    "PHASE_warmup": "warm-up",
    "PHASE_prefix_vision": "iter:vision",
    "PHASE_prefix_build": "iter:build_prefix",
    "PHASE_prefix_prefill": "iter:prefill",
    "PHASE_kv_migration": "iter:kv_migration",
    "PHASE_denoise": "iter:denoise",
    "PHASE_end": "teardown",
}
PHASE_ORDER = [
    "init_one_time",
    "warm-up",
    "iter:vision",
    "iter:build_prefix",
    "iter:prefill",
    "iter:kv_migration",
    "iter:denoise",
    "teardown",
]

# device id -> pipeline stage (from the fingerprint analysis)
VISION = {"0", "4", "8", "12"}
DENOISE = {"9", "10", "11", "17", "18", "19"}


def stage_of(dev):
    return "vision" if dev in VISION else ("denoise" if dev in DENOISE else "prefill")


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def main():
    src, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    rows = list(csv.DictReader(open(src)))
    # order by host timestamp so signposts sit between the ops they bound
    rows.sort(key=lambda r: int(r["HOST START TS"]) if r.get("HOST START TS", "").strip().lstrip("-").isdigit() else 0)

    KEEP = [
        "PHASE",
        "STAGE",
        "OP CODE",
        "OP TYPE",
        "DEVICE ID",
        "CORE COUNT",
        "DEVICE KERNEL DURATION [ns]",
        "DEVICE FW DURATION [ns]",
        "HOST DURATION [ns]",
        "PROGRAM CACHE HIT",
        "GLOBAL CALL COUNT",
    ]
    annotated = []
    phase = "init_one_time"  # everything before the first signpost
    for r in rows:
        if r.get("OP TYPE") == "signpost":
            phase = PHASE_LABEL.get(r.get("OP CODE", "").strip(), r.get("OP CODE", "").strip())
            continue
        dev = r.get("DEVICE ID", "").strip()
        out = {k: r.get(k, "") for k in KEEP}
        out["PHASE"] = phase
        out["STAGE"] = stage_of(dev)
        annotated.append(out)

    # ---- per-op annotated CSV (trimmed) ----
    ops_csv = os.path.join(outdir, "socket_e2e_ops_annotated.csv")
    with open(ops_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEEP)
        w.writeheader()
        w.writerows(annotated)

    # ---- per-phase rollup ----
    KD = "DEVICE KERNEL DURATION [ns]"
    HD = "HOST DURATION [ns]"
    SOCK = {"RecvDirectAsyncDeviceOperation", "SendDirectAsyncDeviceOperation"}
    agg = defaultdict(
        lambda: {
            "ops": 0,
            "dev_kern_us": 0.0,
            "host_us": 0.0,
            "sock_us": 0.0,
            "compute_us": 0.0,
            "devs": set(),
            "perdev": defaultdict(float),
        }
    )
    for r in annotated:
        a = agg[r["PHASE"]]
        a["ops"] += 1
        kd = fnum(r[KD]) / 1000
        a["dev_kern_us"] += kd
        a["host_us"] += fnum(r[HD]) / 1000
        a["devs"].add(r["DEVICE ID"])
        a["perdev"][r["DEVICE ID"]] += kd
        if r["OP CODE"] in SOCK:
            a["sock_us"] += kd
        else:
            a["compute_us"] += kd

    sum_csv = os.path.join(outdir, "socket_e2e_phase_summary.csv")
    cols = ["PHASE", "ops", "chips", "compute_ms", "socket_wait_ms", "busiest_chip_ms", "host_dispatch_ms"]
    with open(sum_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for ph in PHASE_ORDER:
            if ph not in agg:
                continue
            a = agg[ph]
            busiest = max(a["perdev"].values()) / 1000 if a["perdev"] else 0.0
            w.writerow(
                [
                    ph,
                    a["ops"],
                    len(a["devs"]),
                    round(a["compute_us"] / 1000, 3),
                    round(a["sock_us"] / 1000, 3),
                    round(busiest, 3),
                    round(a["host_us"] / 1000, 3),
                ]
            )

    # ---- per-phase x op-type breakdown (compute ms) ----
    pe = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))  # phase -> opcode -> [count, kern_us]
    for r in annotated:
        c = pe[r["PHASE"]][r["OP CODE"]]
        c[0] += 1
        c[1] += fnum(r[KD]) / 1000
    optype_csv = os.path.join(outdir, "socket_e2e_phase_optype.csv")
    with open(optype_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["PHASE", "OP CODE", "count", "device_kernel_ms"])
        for ph in PHASE_ORDER:
            if ph not in pe:
                continue
            for op, (c, us) in sorted(pe[ph].items(), key=lambda x: -x[1][1]):
                w.writerow([ph, op, c, round(us / 1000, 4)])

    # ---- gzip the big per-op CSV so it is committable (<500KB) ----
    import gzip
    import shutil

    with open(ops_csv, "rb") as fi, gzip.open(ops_csv + ".gz", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(ops_csv)

    # ---- console summary ----
    print(f"wrote {ops_csv}.gz ({len(annotated)} ops, gzipped)")
    print(f"wrote {optype_csv}")
    print(f"wrote {sum_csv}")
    print(f"\n{'PHASE':<20}{'ops':>6}{'chips':>6}{'compute_ms':>12}{'socket_ms':>11}{'busiest_ms':>11}{'host_ms':>10}")
    for ph in PHASE_ORDER:
        if ph not in agg:
            continue
        a = agg[ph]
        busiest = max(a["perdev"].values()) / 1000 if a["perdev"] else 0.0
        print(
            f"{ph:<20}{a['ops']:>6}{len(a['devs']):>6}{a['compute_us']/1000:>12.2f}"
            f"{a['sock_us']/1000:>11.2f}{busiest:>11.2f}{a['host_us']/1000:>10.2f}"
        )


if __name__ == "__main__":
    main()

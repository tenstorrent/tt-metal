#!/usr/bin/env python3
"""Print the D2H2H2D bandwidth and latency tables from the stripped CSV.

The CSV that test_oneway_volume writes has one row per stage per run:

    stage,samples,payload_bytes,window_ns,hop_window_ns,bandwidth_gb_per_s,
    messages_per_second,latency_us,total_ns,bytes_per_message,cores,run_id,host_ident

This pivots it into TWO tables, one row per payload size: BANDWIDTH and LATENCY.

EVERY NUMBER IS RECOMPUTED FROM THE RAW COLUMNS -- the derived columns in the file
are checked, not trusted:

    BANDWIDTH, the legs == payload_bytes / hop_window_ns (that leg's own elapsed window)
    BANDWIDTH, e2e      == payload_bytes / window_ns     (completion-bounded throughput)
    LATENCY             == total_ns / samples / 1000     (mean per-message duration)
    payload_bytes       == samples * bytes_per_message   (the integrity check)

Neither it nor the concurrency factor is printed. Both are one division away if you want them,
and S/W == (S/T) x (T/W) closes on the row:

    payload_bytes / total_ns  =  per-core push rate     (S/T)
    total_ns / hop_window_ns  =  messages in flight     (T/W)

Usage:
    ./show_results.py results.csv [more.csv ...]

Pass both roles' files when the sender and receiver wrote separately: the tx side
fills t6->host and host->remote_host, the rx side fills the other two.
"""

import csv
import sys
from collections import defaultdict

H2H = "host->remote_host"

# CSV stage name -> column header. Order here is the column order in the tables.
#
# PLAIN `h2h`, NOT `h2h delivered`. The qualifier existed to distinguish it from `h2h moving`
# in the pair below; with one column there is nothing to distinguish it from, and a qualifier
# that names no alternative reads as a hedge about the number.
STAGES = [
    ("t6->host", "t6->host"),
    (H2H, "h2h"),
    ("remote_host->remote_t6", "remote_host->remote_t6"),
    ("END_TO_END", "end-to-end"),
]


def to_int(s):
    s = (s or "").strip()
    return int(s) if s else None


def to_float(s):
    s = (s or "").strip()
    return float(s) if s else None


def load(paths):
    """Return {(bytes, cores): {stage: {...}}}, a list of problems, and the set of
    (path, stage) pairs whose rate had to fall back to the old total_ns denominator."""
    data = defaultdict(dict)
    problems = []
    stale_window = set()
    for path in paths:
        with open(path, newline="") as fh:
            for lineno, row in enumerate(csv.DictReader(fh), start=2):
                stage = (row.get("stage") or "").strip()
                if not stage:
                    continue
                samples = to_int(row.get("samples"))
                payload = to_int(row.get("payload_bytes"))
                window = to_int(row.get("window_ns"))
                hop_window = to_int(row.get("hop_window_ns"))
                total = to_int(row.get("total_ns"))
                per_msg = to_int(row.get("bytes_per_message"))
                if not samples or not payload:
                    continue  # this process did not measure this stage

                where = f"{path}:{lineno} {stage}"

                if stage == "END_TO_END":
                    bw = payload / window if window else None
                    bw_formula = "payload_bytes/window_ns"
                elif hop_window:
                    bw = payload / hop_window
                    bw_formula = "payload_bytes/hop_window_ns"
                elif window:
                    stale_window.add((path, stage))
                    bw = payload / window
                    bw_formula = "payload_bytes/window_ns [fallback]"
                else:
                    stale_window.add((path, stage))
                    bw = None
                    bw_formula = "no elapsed denominator on this row"

                lat = total / samples / 1000.0 if samples else None
                den = window if stage == "END_TO_END" else (hop_window or window)
                msgs = samples * 1e9 / den if den else None

                file_bw = to_float(row.get("bandwidth_gb_per_s"))
                mismatch = bw is not None and file_bw is not None and abs(bw - file_bw) > 1e-6 * max(1.0, abs(bw))

                if (
                    mismatch
                    and hop_window is None
                    and stage != "END_TO_END"
                    and total
                    and file_bw is not None
                    and abs(payload / total - file_bw) <= 1e-6 * max(1.0, payload / total)
                ):
                    mismatch = False
                if mismatch:
                    problems.append(f"{where}: bandwidth column {file_bw:.6f} != " f"{bw_formula} {bw:.6f}")
                # Check 2: the file's latency matches ours.
                file_lat = to_float(row.get("latency_us"))
                if lat is not None and file_lat is not None and abs(lat - file_lat) > 1e-3:
                    problems.append(f"{where}: latency column {file_lat:.3f} != " f"total_ns/samples/1000 {lat:.3f}")
                # Check 2b: and the messages/second column, where the file carries one. Absent on
                # files written before 2026-09-07, which is not a defect -- .get() returns None
                # and the check does not run.
                file_msgs = to_float(row.get("messages_per_second"))
                if msgs is not None and file_msgs is not None and abs(msgs - file_msgs) > 1e-3 * max(1.0, msgs):
                    problems.append(
                        f"{where}: messages_per_second column {file_msgs:.1f} != "
                        f"samples x 1e9 / denominator {msgs:.1f}"
                    )
                if bw is not None and msgs is not None and per_msg:
                    closed = msgs * per_msg / 1e9
                    if abs(bw - closed) > 1e-6 * max(1.0, bw):
                        problems.append(
                            f"{where}: bandwidth {bw:.6f} != messages_per_second x "
                            f"bytes_per_message/1e9 {closed:.6f}"
                        )
                if per_msg and payload != samples * per_msg:
                    problems.append(
                        f"{where}: payload_bytes {payload} != samples {samples} "
                        f"x bytes_per_message {per_msg} "
                        f"(= {samples * per_msg}, off by {payload - samples * per_msg})"
                    )
                cores = to_int(row.get("cores"))
                if total and window and cores:
                    occ = total / (window * cores)
                    if occ > 1.0:
                        problems.append(
                            f"{where}: occupancy {occ:.2f} > 1 -- total_ns exceeds "
                            f"window_ns x cores; window or samples are wrong"
                        )

                key = (per_msg if per_msg else 0, cores if cores else 0)
                prev = data[key].get(stage)
                if prev and (prev["bw"], prev["lat"]) != (bw, lat):
                    problems.append(f"{where}: repeat run at {key[0]} B / {key[1]} cores, "
                                    f"showing the last one")
                data[key][stage] = {"bw": bw, "lat": lat}
    return data, problems, stale_window


def table(title, formula, data, field, fmt):
    stage_names = [disp for _, disp in STAGES]
    widths = [max(12, len(n) + 2) for n in stage_names]
    bw_col = 12

    print(f"\n{title}")
    print(f"  {formula}\n")
    header = "bytes".rjust(bw_col) + "cores".rjust(7) + "".join(n.rjust(w) for n, w in zip(stage_names, widths))
    print("  " + header)
    print("  " + "-" * len(header))
    for key in sorted(data, key=lambda k: (k[1], k[0])):
        size, cores = key
        cells = []
        for stage_key, _ in STAGES:
            v = data[key].get(stage_key, {}).get(field)
            cells.append(("-" if v is None else format(v, fmt)))
        print("  " + str(size).rjust(bw_col) + str(cores).rjust(7) + "".join(c.rjust(w) for c, w in zip(cells, widths)))


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    data, problems, stale_window = load(argv[1:])
    if not data:
        print("no rows with data -- did both roles write their CSV?", file=sys.stderr)
        return 1

    legs = (
        "payload/window_ns [FALLBACK -- no hop_window_ns in the file(s); too wide by " "pipeline fill and drain]"
        if stale_window
        else "payload/hop_window_ns"
    )
    table(
        "BANDWIDTH (GB/s)   aggregate bytes over one elapsed interval",
        f"legs = {legs}   end-to-end = payload/window_ns",
        data,
        "bw",
        ".3f",
    )
    table("LATENCY (us)   mean per-message duration at that stage", "= total_ns / samples / 1000", data, "lat", ".2f")

    print()
    print("  note: total_ns is RESIDENCE -- per-message durations summed over every core, so it")
    print("        exceeds elapsed time by the number of messages in flight. It is the latency")
    print("        numerator above and nothing else. payload_bytes/total_ns is a PER-CORE push")
    print("        rate, not a link rate, and it does not become one multiplied by `cores`.")
    if stale_window:
        print()
        print(f"  FELL BACK to payload/window_ns on {len(stale_window)} (file, stage) pair(s)")
        print("  with no hop_window_ns value -- written before that leg carried a window. Right")
        print("  shape, wrong width: window_ns is the whole run's bracket, so it is wider than a")
        print("  leg's own envelope by that leg's pipeline fill and drain. Rerun for the real")
        print("  span; do NOT quote these as the leg's measured bandwidth:")
        for p, st in sorted(stale_window):
            print(f"    {p}  [{st}]")
    if problems:
        print(f"CHECKS FAILED ({len(problems)}) -- the numbers above do not reproduce:")
        for p in problems:
            print(f"  {p}")
        return 1
    print("checks passed: every printed number reproduces from the raw columns.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

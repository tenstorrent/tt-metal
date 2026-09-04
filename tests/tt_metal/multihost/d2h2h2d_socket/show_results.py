#!/usr/bin/env python3
"""Print the D2H2H2D bandwidth and latency tables from the stripped CSV.

The CSV that test_oneway_volume writes has one row per stage per run:

    stage,samples,payload_bytes,window_ns,bandwidth_gb_per_s,latency_us,
    total_ns,bytes_per_message,cores,run_id,host_ident

This pivots it into the two tables you actually read, one row per payload size.

EVERY NUMBER IS RECOMPUTED FROM THE RAW COLUMNS -- the derived columns in the file
are checked, not trusted:

    per-hop bandwidth  == payload_bytes / total_ns      (bytes over time spent in that hop)
    end-to-end         == payload_bytes / window_ns     (completion-bounded throughput)
    latency_us         == total_ns / samples / 1000
    payload_bytes      == samples * bytes_per_message   (the integrity check)

1 byte/ns == 1 GB/s decimal, so every one of these is one division, no scale factor.

If any of those disagree the row is flagged instead of printed as fact, because a
number nobody can reproduce by hand is not an answer to a pointed question.

Usage:
    ./show_results.py results.csv [more.csv ...]

Pass both roles' files when the sender and receiver wrote separately: the tx side
fills t6->host and host->remote_host, the rx side fills the other two.
"""

import csv
import sys
from collections import defaultdict

# CSV stage name -> column header. Order here is the column order in the tables.
STAGES = [
    ("t6->host", "t6->host"),
    ("host->remote_host", "host->remote_host"),
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
    """Return {bytes_per_message: {stage: {"bw":, "lat":, "row":}}} plus a list of problems."""
    data = defaultdict(dict)
    problems = []
    for path in paths:
        with open(path, newline="") as fh:
            for lineno, row in enumerate(csv.DictReader(fh), start=2):
                stage = (row.get("stage") or "").strip()
                if not stage:
                    continue
                samples = to_int(row.get("samples"))
                payload = to_int(row.get("payload_bytes"))
                window = to_int(row.get("window_ns"))
                total = to_int(row.get("total_ns"))
                per_msg = to_int(row.get("bytes_per_message"))
                if not samples or not payload:
                    continue  # this process did not measure this stage

                where = f"{path}:{lineno} {stage}"

                # Recompute, do not trust.
                #
                # TWO DIFFERENT RATES, AND THE PER-HOP ONE IS NOT payload/window.
                # payload/window is ACHIEVED throughput. Under backpressure every hop moves
                # the same bytes through the same window, so that formula makes all four
                # columns restate the pipeline rate -- a t6->host column that can never say
                # anything about t6->host. Useless.
                #
                # A HOP's rate is payload/total_ns: bytes over the time actually spent inside
                # that hop. That is what the libfabric/mpi-rma tables reported, and it is the
                # number comparable to them (t6->host 3.295, host->host 9.593, ...).
                #
                # END_TO_END keeps payload/window, because the whole path IS the pipeline and
                # its completion-bounded throughput is the real answer there.
                if stage == "END_TO_END":
                    bw = payload / window if window else None
                else:
                    bw = payload / total if total else None
                lat = total / samples / 1000.0 if samples else None

                # Check 1: the file's bandwidth matches ours.
                file_bw = to_float(row.get("bandwidth_gb_per_s"))
                # The csv column is payload/window on every row, so it is the achieved rate.
                # Only END_TO_END prints that, so only END_TO_END cross-checks against it.
                if (stage == "END_TO_END" and bw is not None and file_bw is not None
                        and abs(bw - file_bw) > 1e-6 * max(1.0, abs(bw))):
                    problems.append(f"{where}: bandwidth column {file_bw:.6f} != "
                                    f"payload_bytes/window_ns {bw:.6f}")
                # Check 2: the file's latency matches ours.
                file_lat = to_float(row.get("latency_us"))
                if lat is not None and file_lat is not None and abs(lat - file_lat) > 1e-3:
                    problems.append(f"{where}: latency column {file_lat:.3f} != "
                                    f"total_ns/samples/1000 {lat:.3f}")
                # Check 3: THE INTEGRITY CHECK. Bytes and samples must count the same
                # population. If this fails the row is counting two different things.
                if per_msg and payload != samples * per_msg:
                    problems.append(f"{where}: payload_bytes {payload} != samples {samples} "
                                    f"x bytes_per_message {per_msg} "
                                    f"(= {samples * per_msg}, off by {payload - samples * per_msg})")
                # Sanity bound: time spent inside a stage cannot exceed the window times
                # the number of cores working in it.
                cores = to_int(row.get("cores"))
                if total and window and cores:
                    occ = total / (window * cores)
                    if occ > 1.0:
                        problems.append(f"{where}: occupancy {occ:.2f} > 1 -- total_ns exceeds "
                                        f"window_ns x cores; window or samples are wrong")

                # KEYED ON (bytes, cores), NOT bytes alone. A csv holding a 110-core sweep and
                # a 1-core sweep has both at every payload size, and keying on size alone made
                # one silently overwrite the other and report every row as a duplicate.
                key = (per_msg if per_msg else 0, cores if cores else 0)
                prev = data[key].get(stage)
                if prev and (prev["bw"], prev["lat"]) != (bw, lat):
                    problems.append(f"{where}: repeat run at {key[0]} B / {key[1]} cores, "
                                    f"showing the last one")
                data[key][stage] = {"bw": bw, "lat": lat}
    return data, problems


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
        print("  " + str(size).rjust(bw_col) + str(cores).rjust(7)
              + "".join(c.rjust(w) for c, w in zip(cells, widths)))


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    data, problems = load(argv[1:])
    if not data:
        print("no rows with data -- did both roles write their CSV?", file=sys.stderr)
        return 1

    table("BANDWIDTH (GB/s)",
          "per-hop = payload_bytes / total_ns      end-to-end = payload_bytes / window_ns",
          data, "bw", ".3f")
    table("LATENCY (us)", "= total_ns / samples / 1000", data, "lat", ".2f")

    print()
    if problems:
        print(f"CHECKS FAILED ({len(problems)}) -- the numbers above do not reproduce:")
        for p in problems:
            print(f"  {p}")
        return 1
    print("checks passed: every printed number reproduces from the raw columns.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

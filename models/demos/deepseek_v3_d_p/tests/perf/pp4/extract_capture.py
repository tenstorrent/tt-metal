"""Reduce an ops_perf_results CSV to ONE device's rows, for committing / sharing.

A stage spans 8 chips (single-rank: 32) and the profiler writes a row per (op instance, device). Those
devices run the same program concurrently, so one device's rows carry the whole per-op picture at
1/8th (or 1/32nd) the size -- 7.5 MB -> 996 KB -> 120 KB gzipped, which fits in a repo.

What you keep: everything tt-perf-report and the analyzers here need. `analyze_layer_budget.py` and
`analyze_kv_ramp.py` normalise per device anyway, so they give identical numbers on the subset.
What you lose: cross-device skew (per-device min/max spread). Go back to the full capture for that.

Usage: extract_capture.py <full_ops_perf_results.csv> <out.csv>   (then gzip the output)
"""
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    raise SystemExit(f"no rows in {sys.argv[1]}")
fields = list(rows[0].keys())
devs = sorted({r["DEVICE ID"] for r in rows if (r.get("DEVICE ID") or "").strip()})
keep_dev = devs[0]
# Rows with a blank DEVICE ID are host-side/signpost records; keep them, they carry the signposts
# tt-perf-report uses for --start-signpost/--end-signpost.
kept = [r for r in rows if (r.get("DEVICE ID") or "").strip() in ("", keep_dev)]
w = csv.DictWriter(open(sys.argv[2], "w", newline=""), fieldnames=fields)
w.writeheader()
w.writerows(kept)
print(f"{sys.argv[1]}\n  devices {devs} -> kept {keep_dev}: {len(kept)}/{len(rows)} rows -> {sys.argv[2]}")

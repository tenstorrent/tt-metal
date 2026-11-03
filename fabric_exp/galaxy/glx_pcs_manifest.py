#!/usr/bin/env python3
"""glx_pcs_manifest.py JOB [--node NODE] [--date YYYY-MM-DD]: the breakdown manifest of one frs_glx_rsz.sbatch sampling
job (report/BREAKDOWN_PLAN.md §5.2, "Galaxy samples: what to save"; read by report/scripts/bd_phase2_sets.py galaxy).

One record per sampled run PASS_CONFIG_s_POINT in /data/aagarwal/frs-glx/jobs/JOB/results (samples, rate CSV, logs),
with the job's kept kernel cache ($J/cache_CONFIG) and the tree's source snapshot (results/src_<router>.tgz). Writes
results/pcs_manifest.json and prints a summary (samples, duration, rate per run). No `ab_targets`: no A/B names a
router for the Galaxy points; the ingest picks the busiest router per ERISC."""
import argparse
import csv
import json
import os
import re
import sys

G = os.environ.get("G", "/data/aagarwal/frs-glx")
POINTS = {"ew1024": (2, "BdGlxEW1024", 1024), "ns7": (4, "BdGlxNS7", 1024), "fullring": (5, "BdGlxFullRing4352", 4352)}
CONFIGS = {"main": ("main", "main", "main (kernel v1, flags 0, firmware keepalive)"),
           "v4rxg3": ("new", "v4rxg3", "v4rxg3 (kernel v2 slot, flags 13129088, keepalive 1000)"),
           # the refresh on the final router (2026-11-03): frs_glx_rsz.sbatch NEWCFG=v4txa11, tree v4txa11_1103
           "v4txa11": ("new", "v4txa11", "v4txa11 (kernel v2 slot, flags 13129088, keepalive 1000)")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--node")
    ap.add_argument("--date", default="2026-11-02")
    a = ap.parse_args()
    J = f"{G}/jobs/{a.job}"
    R = f"{J}/results"
    node = a.node
    if not node:  # from the job log's first line
        for f in os.listdir(f"{G}/logs"):
            if f.endswith(f"-{a.job}.log"):
                m = re.match(r"node=(\S+)", open(f"{G}/logs/{f}").readline())
                node = m.group(1) if m else None
    recs = []
    for f in sorted(os.listdir(R)):
        m = re.match(rf"([a-z0-9]+)_({'|'.join(CONFIGS)})_s_(ew1024|ns7|fullring)_samples\.json$", f)
        if not m:
            continue
        p, cfg, pt = m.groups()
        point, test, size = POINTS[pt]
        router, src, build = CONFIGS[cfg]
        label = f"{p}_{cfg}_s_{pt}"
        meta = json.load(open(f"{R}/{f}"))
        # The logs whose compile commands list this point's router builds (bd_map.py keeps only those hashes): the mesh
        # points' routers are compiled by the job's first 2D row, the k_ check (c_mesh then hits the cache and lists
        # none: jobs 169481, 170096); the full ring's by c_ring. Listing c_ring for a mesh point left no candidate image.
        steps = "check|c_mesh" if pt in ("ew1024", "ns7") else "c_ring"
        compile_logs = [f"{R}/{x}" for x in sorted(os.listdir(R)) if re.match(rf"[a-z0-9]+_{cfg}_({steps})\.log$", x)]
        rate = None
        if os.path.exists(f"{R}/{label}.csv"):
            for r in csv.DictReader(open(f"{R}/{label}.csv")):
                rate = 1.35e9 / float(r["avg_packets_per_s"])
        recs.append({
            "point": point, "router": router, "pass": p,
            "samples_prefix": f"{R}/{label}_samples",
            "rate_csv": f"{R}/{label}.csv",
            "cache": f"{J}/cache_{cfg}",
            "logs": [f"{R}/{label}.log", f"{R}/{label}_sampler.log"] + compile_logs,
            "sources": [f"tgz:{R}/src_{src}.tgz"],
            "build": build, "job": f"Slurm {a.job}", "node": node, "date": a.date,
            "test": test, "packet_size_B": size,
        })
        print(f"{label:22} point {point} {router:4} pass {p}: {meta['samples']:>9} samples in {meta['duration_s']:6.1f} s "
              f"({meta['rate_per_s']}/s), {len(meta['targets'])} targets, {sum(meta['errors'].values())} errors; "
              f"rate {rate and round(rate, 1)} cycles/packet")
    missing = [r["samples_prefix"] for r in recs if not os.path.exists(r["samples_prefix"] + ".bin")]
    missing += [r["cache"] for r in recs if not os.path.isdir(r["cache"])]
    missing += [s[4:] for r in recs for s in r["sources"] if not os.path.exists(s[4:])]
    out = f"{R}/pcs_manifest.json"
    json.dump(recs, open(out, "w"), indent=1)
    print(f"wrote {len(recs)} records to {out}; missing files: {sorted(set(missing)) or 'none'}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())

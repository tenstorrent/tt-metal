#!/usr/bin/env python3
"""Collect the measured data the cost model is calibrated against into calib_data.json (committed, small).

Sources (all on exabox NFS):
  * per-op zone profiles, one prefill chunk of 5120 tokens attending a 51,200-token cache, layers 0 (dense) and 3
    (MoE/MSA), 1D fabric, bf4 experts: prefill_profile_results/<stamp>/ops_perf_results_*.csv parsed with
    models/demos/minimax_m3/tests/perf/parse_zone_perf.py --per-device -> pd_*.json (mean over devices is used)
  * 16-stage [2,4] pipeline runner, per-rank per-chunk compute ms (Sep 27 2026, 120-C pod3, #57827 comment):
    runA = chunk 5120 even split, runB = 5120 split 1,1,1,5x5,4x8, runC = 2048 same split; timing_*/rank*.csv
  * the matrix tables (idle TTFT, loaded steady tok/s) of the same runs: run*/table.csv
Usage: python3 collect_calib.py [--zones DIR] [--matrix DIR]
"""
import argparse
import csv
import glob
import json
import os
import statistics as st

ZONES = {  # mesh -> pd json
    "2x4": "pd_mesh2x4.json",
    "8x4": "pd_8x4.json",
    "4x2": "pd_mesh4x2.json",
}
RUNS = {
    "A": dict(chunk=5120, layers="4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3"),
    "B": dict(chunk=5120, layers="1,1,1,5,5,5,5,5,4,4,4,4,4,4,4,4"),
    "C": dict(chunk=2048, layers="1,1,1,5,5,5,5,5,4,4,4,4,4,4,4,4"),
}


def zone_means(path):
    d = json.load(open(path))
    out = {}
    for k, v in d.items():
        if not k.startswith("profiled_chunk/layer"):
            continue
        name = k.split("/", 1)[1]
        vals = list(v.values()) if isinstance(v, dict) else [v]
        out[name] = dict(mean=round(st.mean(vals), 4), max=round(max(vals), 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    local_zones = os.path.join(os.environ.get("M3SIM_DATA", os.path.join(here, "data")), "zones")
    ap.add_argument(
        "--zones", default=local_zones if os.path.isdir(local_zones) else "/data/philei/m3_traffic_sim/data/zones"
    )
    ap.add_argument("--matrix", default="/data/philei/notes/m3_matrix_2026-09-27_c_pod3")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_data.json"))
    a = ap.parse_args()
    out = dict(
        zones={},
        pipeline={},
        tables={},
        meta=dict(
            zones_condition="chunk 5120, cached 51200, harness capacity 56320, 1D fabric, bf4 experts, mean over devices",
            pipeline_condition="16 stages x [2,4], 2D fabric, PREFILL_SYNC_PER_CHUNK=1, capacity = cached + 51200, bf16 index_k",
            hosts="bh-glx-120-c05u02,c05u08,c04u08,c04u02 (Sep 27 2026)",
        ),
    )
    for mesh, f in ZONES.items():
        p = os.path.join(a.zones, f)
        if os.path.exists(p):
            out["zones"][mesh] = zone_means(p)
    for run, meta in RUNS.items():
        rows, cells = [], []
        recs = [json.loads(l) for l in open(f"{a.matrix}/run{run}/results_16stage.jsonl") if l.strip()]
        for d in sorted(glob.glob(f"{a.matrix}/run{run}/timing_*"), key=lambda x: int(x.split("_c")[-1])):
            cached = int(d.split("_c")[-1])
            per_rank = []
            for r in range(16):
                per_rank.append({int(x[1]): float(x[3]) for x in csv.reader(open(f"{d}/rank{r}.csv"))})
            rows.append(dict(cached=cached, rank_ms=[round(st.median(list(v.values())), 2) for v in per_rank]))
            # segment the chunk stream into cells: per cached level the runner restarts at chunk 0; each cell is
            # its idle iterations (c_first..c_last) followed by its loaded block (total_chunks, contiguous)
            nxt = 0
            for rec in recs:
                if rec.get("cached") != cached:
                    continue
                if rec.get("mode") == "idle":
                    nxt = rec["c_last"] + 1
                    continue
                n_ch, users, C = rec["chunks_per_req"], rec["users"], rec["chunk"]
                idx = range(nxt, nxt + rec["total_chunks"])
                nxt += rec["total_chunks"]
                # producer interleaves users round-robin at chunk granularity: position = (i // users) % n_ch
                pos_ms = [[[] for _ in range(n_ch)] for _ in range(16)]
                for i, c in enumerate(idx):
                    p = (i // users) % n_ch
                    for r in range(16):
                        if c in per_rank[r]:
                            pos_ms[r][p].append(per_rank[r][c])
                cells.append(
                    dict(
                        cached=cached,
                        new=rec["new"],
                        users=users,
                        n_ch=n_ch,
                        pos_ms=[[round(st.median(x), 2) if x else None for x in pr] for pr in pos_ms],
                        period_ms=rec.get("chunk_period_ms_median"),
                    )
                )
        out["pipeline"][run] = dict(meta, rows=rows, cells=cells)
        tab = []
        for r in csv.DictReader(open(f"{a.matrix}/run{run}/table.csv")):
            tab.append({k: (float(v) if v not in ("", "x") else None) for k, v in r.items()})
        out["tables"][run] = tab
    json.dump(out, open(a.out, "w"), indent=0)
    print("wrote", a.out, {k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()

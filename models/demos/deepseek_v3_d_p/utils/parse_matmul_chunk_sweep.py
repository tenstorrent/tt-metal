# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Pick the fastest program config per matmul from a test_glm_matmul_chunk_sweep tracy run.

usage: python -m models.demos.deepseek_v3_d_p.utils.parse_matmul_chunk_sweep <ops_perf_results.csv> <catalog.json> \
           <out.json>

The sweep runs on one chip, so rows are in host order: each `SWEEP|layout|chunk|name|idx` signpost is followed
by that candidate's ITERS matmuls. Per candidate: the fastest DEVICE KERNEL DURATION over its iterations and
the core count. Per matmul: the fastest candidate whose output passed PCC, the TTNN-default row for
reference, and the compute-roofline utilization (HiFi2 32 / HiFi4 64 cycles per tile, 1.35 GHz).
"""

import json
import math
import sys

import pandas as pd

K = "DEVICE KERNEL DURATION [ns]"
FIDELITY_CYCLES = {"HiFi2": 32, "HiFi4": 64}


def parse(csv_path, catalog_path):
    df = pd.read_csv(csv_path, low_memory=False)
    catalog = json.load(open(catalog_path))
    times = {}
    tag = None
    for typ, code, dur, cores in zip(df["OP TYPE"], df["OP CODE"], df[K], df["CORE COUNT"]):
        code = str(code)
        if typ == "signpost":
            tag = code if code.startswith("SWEEP|") else None
            continue
        if tag is None or "Matmul" not in code or pd.isna(dur):
            continue
        t = times.setdefault(tag, {"ns": [], "cores": int(cores) if not pd.isna(cores) else 0})
        t["ns"].append(float(dur))
    per_mm = {}
    for tag, entry in catalog.items():
        spec = entry["spec"]
        rec = per_mm.setdefault(spec["name"], {"spec": spec, "cands": []})
        t = times.get(tag)
        rec["cands"].append(
            dict(
                tag=tag,
                cfg=entry["cfg"],
                out_mem=entry["out_mem"],
                act_mem=entry["act_mem"],
                out_dtype=entry["out_dtype"],
                status=entry["status"],
                us=min(t["ns"]) / 1e3 if t and t["ns"] else None,
                cores=t["cores"] if t else None,
            )
        )
    result = {}
    for name, rec in per_mm.items():
        s = rec["spec"]
        tiles = s["z"] * math.ceil(s["m"] / 32) * (s["k"] // 32) * (s["n"] // 32)
        cyc = FIDELITY_CYCLES["HiFi4" if name == "moe.gate" else "HiFi2"]
        ok = [c for c in rec["cands"] if c["status"] == "ok" and c["us"]]
        best = min(ok, key=lambda c: c["us"]) if ok else None
        default = next((c for c in rec["cands"] if c["cfg"]["kind"] == "default" and c["status"] == "ok"), None)

        def util(c):
            return round(tiles * cyc / max(c["cores"], 1) / 1.35 / (c["us"] * 1e3) * 100, 1) if c and c["us"] else None

        result[name] = dict(
            spec=s,
            best=best,
            default=default,
            best_util=util(best),
            n_ok=len(ok),
            n_total=len(rec["cands"]),
            speedup=round(default["us"] / best["us"], 3) if best and default and default["us"] else None,
        )
    return result


if __name__ == "__main__":
    res = parse(sys.argv[1], sys.argv[2])
    json.dump(res, open(sys.argv[3], "w"), indent=1)
    for name, r in res.items():
        b, d = r["best"], r["default"]
        print(
            f"{name:22s} best {b['us'] if b else float('nan'):8.1f} us {b['cores'] if b else 0:4d} cores "
            f"{(b or {}).get('cfg', {}).get('kind', '-'):9s} out {(b or {}).get('out_mem', '-'):4s} | default "
            f"{d['us'] if d else float('nan'):8.1f} us | util {r['best_util']}% | ok {r['n_ok']}/{r['n_total']}"
        )

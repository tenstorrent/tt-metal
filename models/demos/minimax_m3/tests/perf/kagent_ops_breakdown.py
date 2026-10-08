# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-op device vs host breakdown of the profiled request from a tracy ops CSV (kagent_prefill_bench.py
BENCH_PROFILE=1 under ``python -m tracy -r``). Complements parse_zone_perf.py (zone roll-up) with an op-code view:

  * rows between the ``M3_ZONE_START profiled_chunk`` / ``M3_ZONE_END profiled_chunk`` signposts only
  * one op instance = one GLOBAL CALL COUNT; device time = max over devices of DEVICE KERNEL DURATION (the mesh
    waits for the slowest chip), skew = max - min; host = HOST DURATION (host dispatch of the op)
  * grouped by layer kind (dense / sparse, from the layerNN_* zone the op sits in) and OP CODE, normalised
    per layer instance

Usage: python kagent_ops_breakdown.py <ops_perf_results_*.csv> [--json out.json]
"""

import argparse
import json
import re
import sys

import pandas as pd

DUR = "DEVICE KERNEL DURATION [ns]"
HOST = "HOST DURATION [ns]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--json")
    a = ap.parse_args()
    df = pd.read_csv(a.csv, low_memory=False)
    cols = df.columns
    need = ["OP CODE", "OP TYPE", "DEVICE ID", "GLOBAL CALL COUNT", DUR]
    for c in need:
        assert c in cols, f"missing column {c}"
    in_chunk = False
    n_chunks = 0
    layer = None
    recs = []  # (layer_name, op_code, call_count, device, dur_ns, host_ns)
    layer_re = re.compile(r"M3_ZONE_(START|END) (layer\d+_(dense|sparse))")
    for row in df.itertuples(index=False):
        code = str(row[cols.get_loc("OP CODE")])
        if row[cols.get_loc("OP TYPE")] == "signpost":
            if "M3_ZONE_START profiled_chunk" in code:
                in_chunk = True
                n_chunks += 1
            elif "M3_ZONE_END profiled_chunk" in code:
                in_chunk = False
            m = layer_re.search(code)
            if m:
                layer = m.group(2) if m.group(1) == "START" else None
            continue
        if not in_chunk:
            continue
        dur = row[cols.get_loc(DUR)]
        host = row[cols.get_loc(HOST)] if HOST in cols else float("nan")
        recs.append(
            (
                layer or "outside_layers",
                code,
                row[cols.get_loc("GLOBAL CALL COUNT")],
                row[cols.get_loc("DEVICE ID")],
                dur,
                host,
            )
        )
    r = pd.DataFrame(recs, columns=["layer", "op", "call", "dev", "dur", "host"])
    # GLOBAL CALL COUNT = per-op base + device id: one mesh op instance = (op, call - device)
    r["call"] = r["call"] - r["dev"]
    inst = r.groupby(["layer", "op", "call"]).agg(dev_max=("dur", "max"), dev_min=("dur", "min"), host=("host", "max"))
    inst = inst.reset_index()
    inst["kind"] = inst["layer"].str.extract(r"_(dense|sparse)$")[0].fillna("other")
    # a layer name repeats across the request's chunks (chunk 4096: 2 chunks): numbers are per layer NAME, summed
    # over the request's chunks
    out = {}
    for kind, g in inst.groupby("kind"):
        nl = g["layer"].nunique()
        agg = g.groupby("op").agg(
            n=("call", "count"), dev_ms=("dev_max", "sum"), skew_ms=("dev_min", "sum"), host_ms=("host", "sum")
        )
        agg["skew_ms"] = agg["dev_ms"] - agg["skew_ms"]
        agg = agg / 1e6
        agg["n"] = agg["n"] * 1e6 / nl
        # per layer of ONE request (the capture may hold several profiled requests)
        agg[["n", "dev_ms", "skew_ms", "host_ms"]] /= max(1, n_chunks)
        agg[["dev_ms", "skew_ms", "host_ms"]] /= nl
        agg = agg.sort_values("dev_ms", ascending=False)
        tot = agg[["n", "dev_ms", "host_ms"]].sum()
        print(
            f"\n== {kind}: {nl} layer(s), {n_chunks} profiled request(s); per layer of one request (sum over its chunks): {tot['n']:.0f} ops, "
            f"device {tot['dev_ms']:.2f} ms (max over chips, sum of op kernels); tracy HOST DURATION (C++ enqueue, summed over chips) {tot['host_ms']:.2f} ms"
        )
        print(f"{'op':<58} {'ops/L':>6} {'dev ms':>8} {'skew ms':>8} {'host ms':>8} {'dev %':>6}")
        for op, rr in agg.iterrows():
            print(
                f"{op[:58]:<58} {rr['n']:6.1f} {rr['dev_ms']:8.3f} {rr['skew_ms']:8.3f} {rr['host_ms']:8.3f} "
                f"{100 * rr['dev_ms'] / tot['dev_ms']:6.1f}"
            )
        out[kind] = {"layers": nl, "per_layer": agg.reset_index().to_dict(orient="records"), "totals": tot.to_dict()}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

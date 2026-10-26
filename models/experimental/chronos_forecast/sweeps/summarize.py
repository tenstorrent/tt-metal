# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Summarize a Chronos device profile between signposts, grouped by op + shape."""

import glob
import sys

import pandas as pd

path = sys.argv[1]
if not path.endswith(".csv"):
    path = sorted(glob.glob(f"{path}/**/ops_perf_results_*.csv", recursive=True))[-1]
start = sys.argv[2] if len(sys.argv) > 2 else "chronos_device_forward_start"
stop = sys.argv[3] if len(sys.argv) > 3 else "chronos_device_forward_stop"
df = pd.read_csv(path)
df = df.sort_values("HOST START TS").reset_index(drop=True)
sp = df[df["OP TYPE"] == "signpost"]
i0 = sp[sp["OP CODE"] == start].index[0]
i1 = sp[sp["OP CODE"] == stop].index[0]
d = df.iloc[i0 + 1 : i1]
d = d[d["OP TYPE"] == "tt_dnn_device"].copy()
d["us"] = d["DEVICE KERNEL DURATION [ns]"] / 1000.0
d["gap_us"] = d["OP TO OP LATENCY [ns]"].fillna(0) / 1000.0


def shape(r, k):
    return "x".join(
        str(r[f"INPUT_{k}_{a}_PAD[LOGICAL]"]).split("[")[0] for a in "WZYX" if f"INPUT_{k}_{a}_PAD[LOGICAL]" in r
    )


d["key"] = d.apply(lambda r: f"{r['OP CODE']} {shape(r, 0)} | {shape(r, 1)} c{r['CORE COUNT']}", axis=1)
g = d.groupby("key").agg(n=("us", "size"), total_ms=("us", lambda s: s.sum() / 1000), avg_us=("us", "mean"))
g = g.sort_values("total_ms", ascending=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 140)
pd.set_option("display.max_rows", 200)
print(path)
print(g.head(40).to_string())
print(f"\nops={len(d)} device_ms={d['us'].sum() / 1000:.1f} gaps_ms={d['gap_us'].sum() / 1000:.1f}")

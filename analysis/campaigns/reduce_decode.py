#!/usr/bin/env python3
"""T2.8 decode reduction: per run and position, SDPA decode DEVICE KERNEL DURATION (ops CSV, invocation order = DEC_POS order x DEC_ITERS) and delivered KV GB/s."""
import glob, os, re
from pathlib import Path
import pandas as pd

# Two layouts (PORTABLE_CONTRACT.md): this file in $WORK/handoff/revamp/data/bh_zones, DD that directory;
# or copied into $TTM/analysis/campaigns of a checkout, where DD falls back to $PWD, never the repo tree.
_SD = Path(__file__).resolve().parent
_REPO = next((p for p in (_SD, *_SD.parents) if (p / "ttnn").is_dir() and (p / "tt_metal").is_dir()), None)
DD = os.environ.get("DD") or str(Path.cwd() if _REPO else _SD)
BYTES = {"bfp8_b": 1088 / 1024, "bfloat16": 2.0}
rows = []
for f in sorted(glob.glob(f"{DD}/t28_*_ops_perf_results.csv")):
    tag = os.path.basename(f).replace("_ops_perf_results.csv", "")
    prov = open(f"{DD}/{tag}.csv").readline()
    env = dict(re.findall(r"(\w+)=(\S+)", prov.split("env ")[1].split(" mode")[0]))
    b = int(env["b"])
    nkv = int(env["nkv"])
    d = int(env["d"])
    iters = int(env["iters"])
    kvd = env["kv_dtype"]
    block = int(env["block"])
    positions = [int(x) for x in env["pos"].split(",")]
    o = pd.read_csv(f, low_memory=False)
    dur = [c for c in o.columns if c.startswith("DEVICE KERNEL DURATION")][0]
    s = o[o["OP CODE"].astype(str).str.contains("SdpaDecode|ScaledDotProductAttentionDecode", regex=True)].reset_index(
        drop=True
    )
    for pi, pos in enumerate(positions):
        g = s.iloc[pi * iters : (pi + 1) * iters]
        if len(g) < iters:
            continue
        ns = g[dur].astype(float)
        kv_len = ((pos + 1 + block - 1) // block) * block  # cache positions read, rounded up to the page block
        kv_bytes = b * nkv * kv_len * d * 2 * BYTES[kvd]
        med = ns.iloc[1:].median() if iters >= 3 else ns.median()
        rows.append(
            dict(
                run=tag,
                batch=b,
                grid=env["grid"],
                kv_dtype=kvd,
                position=pos,
                kv_len_read=kv_len,
                kv_bytes=int(kv_bytes),
                dur_ns_iter0=ns.iloc[0],
                dur_ns_iter1=ns.iloc[1] if len(ns) > 1 else float("nan"),
                dur_ns_iter2=ns.iloc[2] if len(ns) > 2 else float("nan"),
                dur_ns_median_excl_first=med,
                kv_GBps=kv_bytes / med,
                cores=int(g["CORE COUNT"].iloc[0]) if "CORE COUNT" in g else -1,
                attributes=str(g["ATTRIBUTES"].iloc[0])[:300],
            )
        )
out = pd.DataFrame(rows)
out.to_csv(f"{DD}/decode_sweep_table.csv", index=False)
pd.set_option("display.width", 250)
print(out.drop(columns=["attributes"]).to_string(index=False) if len(out) else "no rows")

"""Summarise program/layer-attn/run_m_*.log RES lines into markdown tables (task layer-attn). Usage: python summarize.py [LOGDIR]"""
import glob
import json
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "/mnt/tt-data/ssinghal/hy4-logs/program/layer-attn"
rows = []
for f in sorted(glob.glob(f"{D}/run_m_*.log")):
    tag = os.path.basename(f)[4:-4]
    for line in open(f, errors="ignore"):
        if line.startswith("RES "):
            r = json.loads(line[4:])
            r["tag"] = tag
            rows.append(r)


def key(r):
    return (r["layer"], r["mode"], r.get("kv"), r.get("users", 1), r.get("v1", False))


print("## prefill (chunk of 512 tokens, 4x8 mesh, same chunk on the 4 rows)\n")
print(
    "| layer | KV | chunk (p0) | PCC out | max abs diff | ref abs max | idx overlap | traced ms (last chunk, layer total) |"
)
print("|---|---|---|---|---|---|---|---|")
for r in sorted([r for r in rows if r["mode"] == "prefill"], key=lambda r: (r["layer"], r["kv"], r["chunk"])):
    print(
        f"| {r['layer']} | {r['kv']} | {r['chunk']} ({r['p0']}) | {r['pcc']} | {r['max_abs']} | {r['ref_absmax']} | {r.get('idx_overlap','-')} | {r.get('traced_ms','')} |"
    )
print("\n## decode (one token per user; users on mesh rows; context built by the prefill path)\n")
print("| layer | KV | kernel | users | pos | row | PCC out | max abs diff | idx overlap | traced ms/layer |")
print("|---|---|---|---|---|---|---|---|---|---|")
lat = {}
for r in rows:
    if r["mode"] == "decode" and "traced_ms" in r:
        lat[(r["layer"], r["kv"], r.get("v1", False), r.get("users", 1), r["pos"])] = r["traced_ms"]
for r in sorted(
    [r for r in rows if r["mode"] == "decode" and "pcc" in r],
    key=lambda r: (r["layer"], r["kv"], r["v1"], r["users"], r["pos"], r["row"]),
):
    k = (r["layer"], r["kv"], r["v1"], r["users"], r["pos"])
    print(
        f"| {r['layer']} | {r['kv']} | {'split(v1)' if r['v1'] else 'stock(v0)'} | {r['users']} | {r['pos']} | {r['row']} | {r['pcc']} | {r['max_abs']} | {r.get('idx_overlap','-')} | {lat.get(k,'')} |"
    )

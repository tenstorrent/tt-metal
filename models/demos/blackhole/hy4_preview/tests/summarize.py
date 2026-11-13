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
            if "mode" in r:
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


print("\n## compact table (worst case over the chunks / positions / rows of each cell)\n")
print(
    "| layer (index source) | mode | KV | kernel | users | min PCC out | max abs diff | min idx overlap | traced ms / layer |"
)
print("|---|---|---|---|---|---|---|---|---|")
agg = {}
for r in rows:
    if "pcc" not in r or "wbfp8" in r["tag"]:
        continue
    k = (r["layer"], r["mode"], r["kv"], r.get("users", 1), r.get("v1", False))
    a = agg.setdefault(k, {"min": 9, "maxabs": 0, "ovl": 9, "lat": []})
    a["min"] = min(a["min"], r["pcc"])
    a["maxabs"] = max(a["maxabs"], r["max_abs"])
    if "idx_overlap" in r:
        a["ovl"] = min(a["ovl"], r["idx_overlap"])
for r in rows:
    if "traced_ms" in r and "wbfp8" not in r["tag"]:
        k = (r["layer"], r["mode"], r["kv"], r.get("users", 1), r.get("v1", False))
        agg[k]["lat"].append((r.get("pos", r.get("p0")), r["traced_ms"]))
names = {0: "0 (full, dense MLP)", 1: "1 (full)", 2: "2 (shared, ids from layer 1)"}
for k, a in sorted(agg.items()):
    L, mode, kv, users, v1 = k
    lat = ", ".join(f"{('pos ' if mode == 'decode' else 'p0 ')}{p}: {t}" for p, t in a["lat"])
    print(
        f"| {names[L]} | {mode} | {kv} | {'split v1' if v1 else 'stock v0'} | {users} | {a['min']} | {a['maxabs']} | {a['ovl'] if a['ovl'] < 9 else '-'} | {lat} |"
    )

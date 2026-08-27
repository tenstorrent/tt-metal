"""Emit the measured markdown tables for SFPU_LUT_RETUNE_WORMHOLE.md.

Everything the document states numerically comes from here, so the prose, the tables and
the figures cannot drift apart -- they all read curves_{main,retuned}.json and
perf_{main,retuned}.json.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
D = {t: json.load(open(os.path.join(HERE, f"curves_{t}.json"))) for t in ("main", "retuned")}
SEGS = D["main"]["segments"]
OPS = ("sigmoid_appx", "tanh", "gelu_appx")


def segstats(tag, op, i):
    edges = SEGS[op]
    lo, hi = edges[i], edges[i + 1]
    last = i == len(edges) - 2
    e = [abs(p[1] - p[2]) for p in D[tag]["data"][op]
         if lo <= p[0] <= hi and (p[0] < hi or last)]
    x = [p[0] for p in D[tag]["data"][op]
         if lo <= p[0] <= hi and (p[0] < hi or last)]
    worst = max(zip(e, x))
    return {"max": worst[0], "at": worst[1], "mean": sum(e) / len(e), "n": len(e)}


def overall(tag, op):
    return max(segstats(tag, op, i)["max"] for i in range(len(SEGS[op]) - 1))


def label(op, i):
    edges = SEGS[op]
    return (f"`[{edges[i]:g}, {edges[i+1]:g})`" if i < len(edges) - 2
            else f"`[{edges[i]:g}, ∞)`")


print("<!-- SUMMARY -->")
print("| kernel | max\\|err\\| main | max\\|err\\| retuned | factor |")
print("|--------|---------------:|------------------:|-------:|")
for op in OPS:
    a, b = overall("main", op), overall("retuned", op)
    print(f"| `{op}` | {a:.6f} | **{b:.6f}** | **{a/b:.2f}×** |")

for op in OPS:
    print(f"\n<!-- SEGMENTS {op} -->")
    print("| `|x|` | max\\|err\\| main | retuned | mean\\|err\\| main | retuned |")
    print("|-----|---------------:|--------:|----------------:|--------:|")
    for i in range(len(SEGS[op]) - 1):
        a, b = segstats("main", op, i), segstats("retuned", op, i)
        same = abs(a["max"] - b["max"]) <= 1e-9 * a["max"]
        mx = f"{b['max']:.6f}" if same else f"**{b['max']:.6f}**"
        print(f"| {label(op, i)} | {a['max']:.6f} | {mx} | {a['mean']:.6f} | {b['mean']:.6f} |"
              + ("  <!-- unchanged -->" if same else ""))
    a, b = overall("main", op), overall("retuned", op)
    print(f"| **overall** | **{a:.6f}** | **{b:.6f}** | | |")

try:
    P = {t: json.load(open(os.path.join(HERE, f"perf_{t}.json"))) for t in ("main", "retuned")}
except FileNotFoundError as exc:
    print(f"\n<!-- PERF unavailable: {exc.filename} -->")
else:
    print("\n<!-- PERF -->")
    print("| op | dest_acc | main, 4 sessions | retuned | inside main's own range? |")
    print("|----|----------|-----------------:|--------:|:------------------------:|")
    for k in sorted(P["main"]):
        op, da = k.split("|")
        m, n = P["main"][k], P["retuned"][k]
        rng = f"{m['min']}..{m['max']}" if m["min"] != m["max"] else f"{m['min']}"
        got = f"{n['min']}..{n['max']}" if n["min"] != n["max"] else f"{n['min']}"
        inside = "yes" if m["min"] <= n["mean"] <= m["max"] else f"NO ({n['mean']-m['mean']:+.1f})"
        print(f"| `{op}` | {da} | {rng} | {got} | {inside} |")
    spread = max(v["max"] - v["min"] for v in P["main"].values())
    print(f"\n<!-- main sessions: {list(P['main'].values())[0]['sessions']}, "
          f"retuned sessions: {list(P['retuned'].values())[0]['sessions']}; "
          f"widest within-state spread on main: {spread} cycles -->")

print("\n<!-- POINTS -->")
for op in OPS:
    n = sum(segstats("main", op, i)["n"] for i in range(len(SEGS[op]) - 1))
    print(f"<!-- {op}: {n} samples -->")

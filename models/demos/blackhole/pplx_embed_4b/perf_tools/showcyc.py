import glob
import json

B = "."
by = {}
for f in sorted(glob.glob(f"{B}/cyc_*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        by.setdefault(r["name"], {})[r["flag"]] = r["r"]
order = [a["name"] for a in json.load(open(f"{B}/arms_all.json"))]
wall = {}
for l in open(f"{B}/log_serial.txt"):
    if l.startswith("RES "):
        r = json.loads(l[4:])
        wall[r["name"]] = r
print(f"{'arm':30s} {'off kcyc':>9s} {'on kcyc':>9s} {'delta':>8s} {'stab':>6s}   wallclock")
for n in order:
    d = by.get(n, {})
    if 0 not in d or 1 not in d or not d[0] or not d[1]:
        print(f"{n:30s} (pending)")
        continue
    o, x = d[0]["median_cycles"], d[1]["median_cycles"]
    stab = (d[0]["max_cycles"] - d[0]["min_cycles"]) / o * 100
    w = wall.get(n)
    ws = f"{w['delta_pct']:+6.2f}%" if w else ""
    print(f"{n:30s} {o/1e3:9.1f} {x/1e3:9.1f} {(x-o)/o*100:+7.2f}% {stab:5.1f}%   {ws}")

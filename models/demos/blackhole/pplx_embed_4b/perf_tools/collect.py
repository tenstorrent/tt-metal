import glob
import json

B = "."
cyc = {}
for f in sorted(glob.glob(f"{B}/cyc_*.jsonl")):
    hi = "DE" in f
    for l in open(f):
        r = json.loads(l)
        if not r["r"]:
            continue
        # the high-replay rerun supersedes the first pass for the same arm
        k = r["name"]
        if k in cyc and not hi and 0 in cyc[k] and cyc[k].get("hi"):
            continue
        cyc.setdefault(k, {})[r["flag"]] = r["r"]
        if hi:
            cyc[k]["hi"] = True
wall = {}
for l in open(f"{B}/log_serial.txt"):
    if l.startswith("RES "):
        r = json.loads(l[4:])
        wall[r["name"]] = r
out = {}
for n, d in cyc.items():
    if 0 not in d or 1 not in d:
        continue
    o, x = d[0]["median_cycles"], d[1]["median_cycles"]
    w = wall.get(n, {})
    out[n] = dict(
        off_kcyc=round(o / 1e3, 1),
        on_kcyc=round(x / 1e3, 1),
        delta_pct=round((x - o) / o * 100, 2),
        stab_pct=round((d[0]["max_cycles"] - d[0]["min_cycles"]) / o * 100, 1),
        wall_off_us=round(w.get("us_off", 0), 1) or None,
        wall_delta_pct=round(w["delta_pct"], 2) if w else None,
        pcc_off=round(w["pcc_off"], 5) if w else None,
        pcc_on=round(w["pcc_on"], 5) if w else None,
    )
json.dump(out, open(f"{B}/collected.json", "w"), indent=1, sort_keys=True)
for n in sorted(out):
    r = out[n]
    print(
        f"{n:30s} {r['off_kcyc']:8.1f} {r['on_kcyc']:8.1f} {r['delta_pct']:+7.2f}% stab {r['stab_pct']:4.1f}%  "
        f"pcc {r['pcc_off']}->{r['pcc_on']}"
    )

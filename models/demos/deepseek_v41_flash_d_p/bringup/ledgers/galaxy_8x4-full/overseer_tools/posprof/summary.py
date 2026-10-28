"""Print one position's result: host wall, device total, per-section device ms (max over chips)."""
import glob
import json
import os
import sys

D = "/scratch/dnijemcevic/posprof/results"
for p in sys.argv[1:]:
    f = glob.glob(f"{D}/{os.environ.get('TAG','POS')}{p}_profile.json")
    m = (
        json.load(open(f"{D}/{os.environ.get('TAG','POS')}{p}.json"))["metrics"]
        if os.path.exists(f"{D}/{os.environ.get('TAG','POS')}{p}.json")
        else {}
    )
    if not f:
        print(f"POS {p}: no profile json; files: {os.listdir(D)}")
        continue
    d = json.load(open(f[0]))
    dev = d["device_ms"]
    tot = sum(dev.values())

    def v(k):
        x = m.get(k)
        return x.get("value") if isinstance(x, dict) else x

    print(
        f"POS {p}: chunk {d['chunk']}, device {tot:.1f} ms; profiled run built {v('profiled_new_programs')} programs, plain run {v('plain_new_programs')} (host times not shown: not warm)"
    )
    for k, x in sorted(dev.items(), key=lambda kv: -kv[1]):
        if x > 0:
            print(f"  {k:16} {x:8.2f}")

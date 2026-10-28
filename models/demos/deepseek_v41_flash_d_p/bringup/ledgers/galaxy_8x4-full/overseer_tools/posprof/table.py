"""Per-section table (device ms per chunk, 40 layers; and per layer), op calls / device programs per chunk, wall, for
several tagged runs at one position. Usage: table.py <pos> TAG[=label] ..."""
import collections
import gzip
import json
import re
import sys

D = "/scratch/dnijemcevic/posprof"
pos, tags = sys.argv[1], [t.split("=") for t in sys.argv[2:]]
rows, meta = collections.defaultdict(dict), {}
for t in tags:
    tag, lab = t[0], (t[1] if len(t) > 1 else t[0])
    tag, wtag = (tag.split("/") + [tag])[:2]  # TAG/WALLTAG: the wall from another series (FLAT/FLAT2)
    prof = json.load(open(f"{D}/results/{tag}{pos}_profile.json"))
    ops = json.load(gzip.open(f"{D}/results/{tag}{pos}_ops.json.gz"))
    calls, progs = collections.Counter(), collections.Counter()
    for key, lst in ops.items():
        sec = key.split(".", 1)[1] if "." in key else key
        for o in lst:
            calls[sec] += o.get("calls", 0)
            progs[sec] += o.get("programs", 0)
    for sec, ms in prof["device_ms"].items():
        rows[sec][lab] = (ms, calls[sec], progs[sec])
    wall = None
    try:
        w = open(f"{D}/{wtag}wall{pos}.log", errors="replace").read()
        m = re.findall(r"median ([0-9.]+)", w)
        wall = float(m[-1]) if m else None
    except OSError:
        pass
    meta[lab] = (sum(prof["device_ms"].values()), sum(calls.values()), sum(progs.values()), wall)
labs = [t[1] if len(t) > 1 else t[0].split("/")[0] for t in tags]
print(f"position {pos}: device ms per 5120-token chunk (40 layers) | per layer | op calls per chunk")
print(f"{'section':16}" + "".join(f"{l:>30}" for l in labs))
for sec in sorted(rows, key=lambda s: -max(v[0] for v in rows[s].values())):
    print(
        f"{sec:16}"
        + "".join(
            f"{rows[sec].get(l, (0, 0, 0))[0]:9.2f} {rows[sec].get(l, (0, 0, 0))[0] / 40:7.3f} {rows[sec].get(l, (0, 0, 0))[1]:7d}  "
            for l in labs
        )
    )
print(f"{'TOTAL device':16}" + "".join(f"{meta[l][0]:9.1f} {meta[l][0] / 40:7.2f} {meta[l][1]:7d}  " for l in labs))
print(f"{'host wall':16}" + "".join(f"{(meta[l][3] or float('nan')):9.1f}{'':17}" for l in labs))
print(f"{'wall - device':16}" + "".join(f"{((meta[l][3] or float('nan')) - meta[l][0]):9.1f}{'':17}" for l in labs))

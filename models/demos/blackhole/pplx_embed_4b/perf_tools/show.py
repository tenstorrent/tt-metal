import glob
import json

rows = []
for f in sorted(glob.glob("./log_*.txt")):
    for l in open(f):
        if l.startswith("RES "):
            rows.append(json.loads(l[4:]))
for r in sorted(rows, key=lambda x: x["name"]):
    if "error" in r:
        print(f"{r['name']:32s} ERROR {r['error'][:100]}")
    else:
        print(
            f"{r['name']:32s} off={r['us_off']:8.1f} on={r['us_on']:8.1f} {r['delta_pct']:+6.2f}%  "
            f"pcc {r['pcc_off']:.5f}->{r['pcc_on']:.5f} ident={r['identical']}"
        )
print(f"-- {len(rows)} arms done")

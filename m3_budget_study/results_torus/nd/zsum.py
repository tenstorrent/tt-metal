import json, sys

# usage: zsum.py label=zones.json ...  -> sparse-layer zone max/mean (device ms) + ops
want = ["mlp/dispatch", "mlp/experts_mm", "mlp/combine", "mlp/moe_reduce", "mlp", ""]
for arg in sys.argv[1:]:
    lab, path = arg.split("=", 1)
    z = json.load(open(path))["zones"]
    lay = [k for k in z if k.count("/") == 1 and "sparse" in k][0]
    subs = sorted(k[len(lay) + 1 :] for k in z if k.startswith(lay + "/mlp/") and k.count("/") == 3)
    print(f"{lab} ({lay}) mlp children: {subs}")
    for w in want + [s for s in subs if s not in want]:
        k = lay + ("/" + w if w else "")
        if k in z:
            print(
                f"  {w or 'layer':28s} max {z[k]['ms_max']:7.3f}  mean {z[k]['ms_mean']:7.3f}  min {z[k]['ms_min']:7.3f}"
            )
    c = "profiled_chunk"
    print(f"  {'chunk':28s} max {z[c]['ms_max']:7.3f}  mean {z[c]['ms_mean']:7.3f}")

#!/usr/bin/env python3
"""Markdown tables from results_ops/goodput/*.json (tools/goodput_study.sh)."""
import json, os

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = os.path.join(R, "goodput")


def load(name):
    d = json.load(open(os.path.join(G, name + ".json")))
    out = {}
    for r in d["results"]:
        at = r.get("at") or {}
        out[(r["features"], r["topo"], r["budget"], r["slo"])] = (r["goodput"], at.get("conc"), at.get("ttftP90"))
    return out


def k(x):
    return f"{x / 1000:.1f}k"


def mesh_table(cals, feats):
    rows = [
        "| calibration | features | W | SLO p90 | 16x[2,4] | 16x[4,2] | [4,2]/[2,4] |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for cal, label in cals:
        for f in feats:
            for W in (4096, 8192):
                d = load(f"{cal}_w{W}")
                for slo in (10, 3):
                    a = d[(f, "16x[2,4]", W, slo)][0]
                    b = d[(f, "16x[4,2]", W, slo)][0]
                    rows.append(
                        f"| {label} | {f} | {W} | {slo} s | {k(a)} | {k(b)} | {b / a:.3f} ({100 * (b / a - 1):+.1f}%) |"
                    )
    return "\n".join(rows)


FIX = [
    ("sparse", "sparse -> 70% (target)"),
    ("moe_reduce", "moe_reduce -> 80% (target)"),
    ("moe_reduce_minchip", "moe_reduce -> min-chip eff (imbalance wait removed, kernel as is)"),
    ("combine", "combine -> min-chip eff (imbalance wait removed)"),
    ("experts", "experts -> 70% (target)"),
    ("experts_imb1", "experts: roofline imbalance 1.2 -> 1.0 (--set expertImb=1)"),
    ("dispatch", "dispatch -> 80% (target)"),
    ("dispatch_minchip", "dispatch -> min-chip eff"),
    ("all5", "all 5 (sparse, moe_reduce, combine, experts, dispatch; first-listed values)"),
]


def fix_table():
    base = {W: load(f"fix/base.w{W}") for W in (4096, 8192)}
    hdr = (
        "| fix | "
        + " | ".join(f"{f} W{W} {s}s" for f in ("near", "full") for W in (4096, 8192) for s in (10, 3))
        + " |"
    )
    rows = [hdr, "|---|" + "---:|" * 8]
    rows.append(
        "| base (ours, 16x[2,4]) | "
        + " | ".join(
            k(base[W][(f, "16x[2,4]", W, s)][0]) for f in ("near", "full") for W in (4096, 8192) for s in (10, 3)
        )
        + " |"
    )
    for name, label in FIX:
        cells = []
        for f in ("near", "full"):
            for W in (4096, 8192):
                d = load(f"fix/{name}.w{W}")
                for s in (10, 3):
                    a = base[W][(f, "16x[2,4]", W, s)][0]
                    b = d[(f, "16x[2,4]", W, s)][0]
                    cells.append(f"{100 * (b / a - 1):+.1f}%")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main_cals = [
        ("pavlo", "Pavlo's"),
        ("ours", "ours (both meshes, [4,2] with copies)"),
        ("ours_native", "ours, [4,2] native gather"),
    ]
    print("## main\n" + mesh_table(main_cals, ("near", "full")))
    print("\n## p0p1\n" + mesh_table(main_cals, ("p0p1",)))
    sens = [
        ("ours_native_pavlopipe", "ours native, Pavlo's pipe.ringC"),
        ("ours_prose", "ours, [2,4] prose only, [4,2] with copies"),
        ("ours_prose_native", "ours, [2,4] prose only, [4,2] native"),
    ]
    print("\n## sensitivity\n" + mesh_table(sens, ("near", "full")))
    print("\n## fix\n" + fix_table())
    # operating points for the main table
    print("\n## points (conc, p90) ours_native")
    for W in (4096, 8192):
        d = load(f"ours_native_w{W}")
        for key, v in sorted(d.items()):
            print(key, k(v[0]), v[1], v[2])

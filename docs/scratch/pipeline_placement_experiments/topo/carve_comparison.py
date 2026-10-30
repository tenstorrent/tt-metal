# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The carve-requiring comparison: footprint (topology solver, Auto) against the resolver path (carve planner
then resolver) and the topology solver at slot level, per case. Reads results/<run>/{completeness,
footprint_then_resolver, planner_then_resolver}. Usage: carve_comparison.py <results dir> [group ...]"""
import glob
import os
import re
import sys
from collections import Counter

RES = sys.argv[1] if len(sys.argv) > 1 else "results/rerun_2026-10-27"
GROUPS = sys.argv[2:] or [
    "encodings",
    "sweep",
    "min2",
    "hetero",
    "multi4",
    "designed",
    "repo",
    "multi",
    "rings",
    "multi2",
    "multi3",
]
CELL = re.compile(r"\|\s*([A-Za-z0-9+_-]+(?:\([a-z?]+\))?)\s+(yes|no|cap|lim|BAD)\s*([\d.]+)s")


def rows(path):
    out = {}
    if not os.path.exists(path):
        return out
    for l in open(path).read().splitlines():
        if "|" not in l or l.startswith(("engine", "carve as")) or "warning" in l:
            continue
        head = l.split("|")[0]
        name = re.sub(r"\s+\d+/\d+(\s+\d+ pos|\s+(exact|ports))?\s*$", "", head).strip()
        if name == head.strip():  # slot / plan / planner rows end with "N stages[/ M slots]" instead
            name = re.sub(r"\s+\d+ stages(/\s*\d+ slots)?$", "", name).strip()
        cells = {
            re.sub(r"\([^)]*\)", "", lab).replace("topo-", "topo "): (v, float(t)) for lab, v, t in CELL.findall(l)
        }
        m = re.search(r"planner (yes|no|cap)\s+([\d.]+)s(?: \(\d+ exp, \d+ cyc\))?\s*([a-z0-9-]*)", l)
        if m:
            cells["planner"] = (m.group(1), float(m.group(2)))
            cells["planner_phase"] = (m.group(3), 0.0)
        if "planner n/a" in l:
            cells["planner"] = ("n/a", 0.0)
        out[name] = cells
    return out


def cell(d, name, eng):
    c = d.get(name, {}).get(eng)
    if c is None:
        return "-"
    v, t = c
    return f"{v} {t:.3g}" if v in ("yes", "no") else v


def total(d1, n1, e1, d2, n2, e2):
    a, b = d1.get(n1, {}).get(e1), d2.get(n2, {}).get(e2)
    if not a or a[0] != "yes":
        return "-"
    if not b or b[0] != "yes":
        return f"carve {a[1]:.3g}, then {b[0] if b else '-'}"
    return f"{a[1] + b[1]:.3g}"


def casedefs(tag):
    out = []
    for l in open(f"{RES}/casedefs/{tag}.cases"):
        f = l.rstrip("\n").split("\t")
        shapes = [tuple(map(int, p.split())) for p in f[3].split(";") if p.strip()]
        meshes = [tuple(map(int, p.split())) for p in f[1].split(";") if p.strip()]
        out.append((f[0], meshes, shapes, f[4]))
    return out


def mix(shapes):
    c = Counter(f"{h}x{w}" for h, w in shapes)
    return ", ".join(f"{c[s]}x {s}" if c[s] > 1 else s for s in sorted(c, key=lambda s: (-int(s[0]) * int(s[2]), s)))


for tag in GROUPS:
    cap = 30 if tag == "multi3" else 60
    fp = rows(f"{RES}/footprint_then_resolver/{tag}_footprint_auto_60s.txt")
    fpc = rows(f"{RES}/completeness/{tag}_footprint_auto_compl_{cap}s.txt")
    res_fp = rows(f"{RES}/completeness/{tag}_resolver_after_carve_60s.txt") or rows(
        f"{RES}/footprint_then_resolver/{tag}_resolver_after_carve_60s.txt"
    )
    topo_fp = rows(f"{RES}/completeness/{tag}_topo_auto_on_footprint_slots_60s.txt") or rows(
        f"{RES}/footprint_then_resolver/{tag}_topo_auto_on_footprint_slots_60s.txt"
    )
    pl = rows(f"{RES}/planner_then_resolver/{tag}_planner_resolver_60s.txt")
    plw = rows(f"{RES}/planner_then_resolver/{tag}_wide0_planner_resolver_60s.txt")
    topo_pl = rows(f"{RES}/planner_then_resolver/{tag}_topo_auto_on_planner_slots_60s.txt")
    # for the 1x1 multi-mesh groups the per-slice topology and the resolver run directly too
    direct_topo = rows(f"{RES}/topo_{tag}_{cap}s_sat_unlimited.txt")
    direct_res = rows(f"{RES}/resolver_{tag}_{cap}s.txt")
    if tag == "multi2":
        fp = rows(f"{RES}/footprint_multi2_auto_60s_sat_unlimited.txt")
    print(f"### {tag}\n")
    cols = [
        "footprint Auto",
        "footprint Auto + completeness",
        "resolver after footprint carve",
        "topology Auto after footprint carve",
        "planner (C++ port)",
        "planner, stage-0 rule off",
        "resolver after planner carve",
        "topology Auto after planner carve",
    ]
    if direct_topo:
        cols += ["topology per-slice Auto, direct", "resolver, direct (one slice = one submesh)"]
    print("| Case | Meshes | Stages | Area / slices | Verdict | " + " | ".join(cols) + " |")
    print("|---|---|---|---|---|" + "---|" * len(cols))
    for name, meshes, shapes, kind in casedefs(tag):
        area = sum(h * w for h, w in shapes)
        slices = sum(r * c for r, c in meshes)
        f1, f2 = cell(fp, name, "auto"), cell(fpc, name, "auto")
        yes = [
            x
            for x in (
                f1,
                f2,
                cell(direct_topo, name, "topo auto"),
                cell(direct_res, name, "resolver"),
                cell(pl, name, "planner"),
            )
            if x.startswith("yes")
        ]
        no = [x for x in (f1, f2, cell(direct_res, name, "resolver")) if x.startswith("no")] + (
            [cell(direct_topo, name, "topo auto")] if cell(direct_topo, name, "topo auto").startswith("no") else []
        )
        verdict = "feasible" if yes else ("infeasible" if no or area > slices else "open")
        p1 = cell(pl, name, "planner")
        if p1.startswith("no"):
            p1 = "no: " + pl[name]["planner_phase"][0]
        p2 = cell(plw, name, "planner") if plw else "-"
        if p2.startswith("no"):
            p2 = "no: " + plw[name]["planner_phase"][0]
        vals = [
            f1,
            f2,
            cell(res_fp, name, "resolver-after-carve"),
            cell(topo_fp, name, "topo auto"),
            p1,
            p2,
            cell(pl, name, "resolver") if cell(pl, name, "resolver") != "-" else cell(plw, name, "resolver"),
            cell(topo_pl, name, "topo auto"),
        ]
        if direct_topo:
            vals += [cell(direct_topo, name, "topo auto"), cell(direct_res, name, "resolver")]
        mesh_s = (
            f"{len(meshes)} x {meshes[0][0]}x{meshes[0][1]}"
            if len(set(meshes)) == 1
            else "; ".join(f"{r}x{c}" for r, c in meshes)
        )
        print(
            f"| {name} | {mesh_s} | {mix(shapes)} ({kind}) | {area}/{slices} | {verdict} | " + " | ".join(vals) + " |"
        )
    print()

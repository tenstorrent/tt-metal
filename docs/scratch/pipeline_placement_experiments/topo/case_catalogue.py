# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Every instance in the carve / multi-mesh / slot experiments: what it is, whether it is solvable, and
which engine decided it fastest.

Usage: case_catalogue.py <results dir>   (the 2026-10-27 layout; prints Markdown)

A verdict is
  feasible    some engine returned a solution that passed the harness's independent check
  infeasible  a complete search said "no": Python dfs/dfs2/pos, the resolver, a DFS engine that did not hit
              its call limit, the SAT engine with the conflict budget removed, or (multi-mesh, 1 cable per
              pair, tight) the colour test
  open        nothing decided it within its cap
"no" answers from a SAT engine run under the default conflict budget are never used as proofs.
"""
import glob
import os
import re
import sys
from collections import Counter, defaultdict

RES = sys.argv[1] if len(sys.argv) > 1 else "results/rerun_2026-10-27"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import multi_colour_test as mc  # noqa: E402

CELL = re.compile(r"\|\s*([A-Za-z0-9+_-]+(?:\([a-z?]+\))?)\s+(yes|no|cap|lim|BAD|\?\?\?)\s*>?\s*([\d.]+)s")


def parse_file(path, trusted_no):
    """{case name: {engine label: (verdict, seconds)}} for one result file."""
    out = {}
    try:
        lines = open(path).read().splitlines()
    except FileNotFoundError:
        return out
    for l in lines:
        if "|" not in l or l.startswith("engine") or l.startswith("carve as") or "warning" in l:
            continue
        head = l.split("|")[0]
        name = re.sub(r"\s+\d+/\d+(\s+\d+ pos|\s+(exact|ports))?\s*$", "", head).strip()
        name = re.sub(r"\s+\d+ stages/\s*\d+ slots$", "", name).strip()
        cells = {}
        for label, v, t in CELL.findall(l):
            label = re.sub(r"\([^)]*\)", "", label).replace("topo-", "topo ")
            if v == "???":
                v = "cap"
            # only a SAT "no" depends on the conflict budget; a DFS "no" that is not "lim" is a complete search
            if v == "no" and not trusted_no and ("sat" in label or "auto" in label):
                v = "no?"
            cells[label] = (v, float(t))
        out[name] = cells
    return out


def merge(*parsed):
    m = defaultdict(dict)
    for prefix, d in parsed:
        for name, cells in d.items():
            for label, vt in cells.items():
                m[name][f"{prefix}{label}"] = vt
    return m


def decide(cells, colour=None):
    yes = sorted((t, e) for e, (v, t) in cells.items() if v == "yes")
    # A DFS "no" on a case another engine solved is the 10M call limit reported as failure (the session
    # API does not pass stats.dfs_calls through, so the harness could not mark it "lim"): mark it here.
    if yes:
        for e, (v, t) in list(cells.items()):
            if v == "no" and "dfs" in e:
                cells[e] = ("lim", t)
    no = sorted((t, e) for e, (v, t) in cells.items() if v == "no")
    if yes:
        t, e = yes[0]
        how = f"{e} {t:.3g} s"
        if colour and colour.startswith("feasible"):
            how += "; colour test agrees"
        return "feasible", how, yes
    if colour and colour.startswith("INFEASIBLE"):
        return "infeasible", "colour test (same-colour cable slices on a tight mesh)", []
    if no:
        t, e = no[0]
        return "infeasible", f"{e} proves it in {t:.3g} s", []
    return "open", "every engine capped", []


def fmt(cells, order):
    parts = []
    for e in order:
        if e in cells:
            v, t = cells[e]
            parts.append(f"{v} {t:.3g}" if v in ("yes", "no", "no?") else v)
        else:
            parts.append("-")
    return " | ".join(parts)


def shape_mix(shapes):
    c = Counter(f"{h}x{w}" for h, w in shapes)
    order = sorted(c, key=lambda s: (-int(s[0]) * int(s[2]), s))
    return ", ".join(f"{c[s]}x {s}" if c[s] > 1 else s for s in order)


def read_cases(fname):
    for line in open(os.path.join(HERE, fname)):
        if line.strip():
            name, R, C, sh = line.rstrip("\n").split("\t")
            yield name, int(R), int(C), [tuple(map(int, p.split())) for p in sh.split(";")]


def table_single(title, case_file, py_file, fp_files, extra_proofs=None):
    print(f"### {title}\n")
    order = ["py dfs", "py dfs2", "py pos", "fp sat", "fp dfs"]
    print("| Case | Grid | Stages | Area / slices | Verdict | Decided by | " + " | ".join(order) + " |")
    print("|---|---|---|---|---|---|" + "---|" * len(order))
    m = merge(
        ("py ", parse_file(f"{RES}/{py_file}", True)),
        *[("fp ", parse_file(f"{RES}/{f}", "unlimited" in f or "600s" in f)) for f in fp_files],
    )
    for name, R, C, shapes in read_cases(case_file):
        cells = m.get(name, {})
        area = sum(h * w for h, w in shapes)
        v, how, _ = decide(cells)
        if v == "open" and extra_proofs and name in extra_proofs:
            v, how = "infeasible", extra_proofs[name]
        if area > R * C:
            v, how = "infeasible", f"area {area} > {R*C} slices"
        print(f"| {name} | {R}x{C} | {shape_mix(shapes)} | {area}/{R*C} | {v} | {how} | {fmt(cells, order)} |")
    print()


def table_multi(title, gen, files):
    print(f"### {title}\n")
    order = ["resolver", "topo auto", "topo sat", "topo dfs", "fp auto", "fp sat", "fp dfs"]
    print("| Case | Cables | Colour test | Verdict | Decided by | " + " | ".join(order) + " |")
    print("|---|---|---|---|---|" + "---|" * len(order))
    m = merge(*[(p, parse_file(f"{RES}/{f}", t)) for p, f, t in files])
    for name, n, k, used, ring, cables in gen():
        colour = mc.decide(n, k, used, ring, cables)
        cells = {re.sub(r"\(\w+\)", "", e).strip(): vt for e, vt in m.get(name, {}).items()}
        v, how, _ = decide(cells, colour)
        cab = (
            " ".join(f"m{a}r{ra}-m{b}r{rb}" for a, ra, _, b, rb, _ in cables)
            if len(cables) <= 4
            else f"{len(cables)} cables, {k} per pair"
        )
        print(f"| {name} | {cab} | {colour} | {v} | {how} | {fmt(cells, order)} |")
    print()


def table_fp_only(title, fname):
    print(f"### {title}\n")
    order = ["sat", "dfs"]
    print("| Case | Area / slices | Positions | Verdict | Decided by | fp sat | fp dfs |")
    print("|---|---|---|---|---|---|---|")
    for l in open(f"{RES}/{fname}").read().splitlines():
        if "|" not in l or l.startswith("carve as"):
            continue
        head = l.split("|")[0]
        mm = re.match(r"(.*?)\s+(\d+)/(\d+)\s+(\d+) pos", head)
        cells = {lab: (v if v != "???" else "cap", float(t)) for lab, v, t in CELL.findall(l)}
        v, how, _ = decide(cells)
        print(f"| {mm.group(1)} | {mm.group(2)}/{mm.group(3)} | {mm.group(4)} | {v} | {how} | {fmt(cells, order)} |")
    print()


def table_rings():
    print(
        "### Single-slice rings on one grid and across two meshes (`all` / `rings`, 60 s cap): every engine applies\n"
    )
    order = ["resolver", "topo auto", "topo sat", "topo dfs", "fp auto"]
    print("| Case | Verdict | Decided by | " + " | ".join(order) + " |")
    print("|---|---|---|" + "---|" * len(order))
    m = merge(
        ("", parse_file(f"{RES}/resolver_single_slice_rings_60s.txt", True)),
        ("", parse_file(f"{RES}/topo_single_slice_rings_60s_sat_unlimited.txt", True)),
        ("fp ", parse_file(f"{RES}/footprint_single_slice_rings_auto_60s_sat_unlimited.txt", True)),
    )
    for name, cells in m.items():
        v, how, _ = decide(cells)
        if v == "open" and ("same colour" in name or "odd" in name):
            v, how = "infeasible", "colour: same-colour cable slices (or an odd grid), no Hamiltonian cycle"
        print(f"| {name} | {v} | {how} | {fmt(cells, order)} |")
    print()


def table_slots(title, res_file, topo_file):
    print(f"### {title}\n")
    order = ["resolver", "topo auto", "topo sat", "topo dfs"]
    print("| Case | Verdict | Decided by | " + " | ".join(order) + " |")
    print("|---|---|---|" + "---|" * len(order))
    m = merge(("", parse_file(f"{RES}/{res_file}", True)), ("", parse_file(f"{RES}/{topo_file}", True)))
    for name, cells in m.items():
        cells = {re.sub(r"\(\w+\)", "", e).strip(): vt for e, vt in cells.items()}
        v, how, _ = decide(cells)
        print(f"| {name} | {v} | {how} | {fmt(cells, order)} |")
    print()


if __name__ == "__main__":
    print("## Appendix: every instance, its definition, verdict and fastest engine\n")
    print(
        f"Results directory `{RES}`. Times in seconds; `cap` = wall-clock cap, `lim` = the DFS engine's 10M call "
        'limit (undecided), `no?` = a SAT "no" under the default conflict budget (not used as a proof).\n'
    )
    table_single(
        "Single-mesh carve: the original 17 cases (1x1 stages allowed)",
        "carve_encoding_cases.txt",
        "encodings_python_60s.txt",
        [
            "footprint_encodings_cases_60s_sat_unlimited.txt",
            "footprint_encodings_cases_60s_default_budget.txt",
            "footprint_undecided_600s_sat_unlimited.txt",
        ],
        {
            "7x7: 1x1, 10x 2x2 (only 9 fit)": "lattice count: 10 stages of 2x2 need 10 marked cells, a 7x7 grid has 9",
            "5x5 ring of 25 x 1x1 (odd: none)": "odd grid: no Hamiltonian cycle",
        },
    )
    table_single(
        "Single-mesh carve: heterogeneous random sweep (1x1 to 2x4)",
        "carve_sweep_cases.txt",
        "sweep_python_60s.txt",
        ["sweep_footprint_60s_sat_unlimited.txt"],
    )
    table_single(
        "Single-mesh carve: no stage smaller than 1x2",
        "carve_min2_cases.txt",
        "min2_python_60s.txt",
        ["min2_footprint_60s_sat_unlimited.txt"],
    )
    table_multi(
        "Multi-mesh rings and chains of one-slice stages: the original 18 (`multi2`), 60 s cap",
        mc.multi2,
        [
            ("", "resolver_multi2_60s.txt", True),
            ("", "topo_multi2_60s_sat_unlimited.txt", True),
            ("fp ", "footprint_multi2_60s_sat_unlimited.txt", True),
            ("fp ", "footprint_multi2_60s_default_budget.txt", False),
            ("fp ", "footprint_multi2_auto_60s_sat_unlimited.txt", True),
            ("fp ", "footprint_undecided_600s_sat_unlimited.txt", True),
        ],
    )
    table_multi(
        "Multi-mesh rings and chains of one-slice stages: the wider sweep (`multi3`), 30 s cap",
        mc.multi3,
        [
            ("", "resolver_multi3_30s.txt", True),
            ("", "topo_multi3_30s_sat_unlimited.txt", True),
            ("fp ", "footprint_multi3_30s_sat_unlimited.txt", True),
        ],
    )
    table_fp_only(
        "Multi-mesh heterogeneous rings (`hetero`, footprint only, 60 s cap)", "footprint_hetero_60s_sat_unlimited.txt"
    )
    table_fp_only(
        "Multi-mesh rings with no stage smaller than 1x2 (`multi4`, footprint only, 60 s cap)",
        "footprint_multi4_60s_sat_unlimited.txt",
    )
    table_rings()
    table_slots(
        "Pre-carved slots, 2x2 and 2x4 (`slots`, 60 s cap)",
        "resolver_slots_60s.txt",
        "topo_slots_60s_sat_unlimited.txt",
    )
    table_slots(
        "Pre-carved slots, 1x2 and 2x1 dominoes (`slots2`, 60 s cap)",
        "resolver_slots2_60s.txt",
        "topo_slots2_60s_sat_unlimited.txt",
    )

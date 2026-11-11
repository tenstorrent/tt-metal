#!/usr/bin/env python3
"""Regenerate the cells where the half sync closed block of the matmul pack reads slower than main from matmul_closed_off_rows.csv (the measured rows: ABBA x2 and 3 pure-shift arms per side).

A key (fidelity, A, B, output, rt, ct, kt, throttle, packer L1 accumulate) with a closed-slow row takes the per-tile pack (matmul_pack_closed_off_cells / _PACK_CLOSED_OFF) when the per-tile pack is not slower in
any measured row of the key; otherwise the closed-slow rows (with their accumulate mode and nb) take main's whole form (matmul_grid_main_cells / _GRID_MAIN_ROWS).
A row with a form column (pt or main) takes that form (the measured fastest form not slower than main); rows without one follow the rule above.
usage: gen_matmul_closed_off_cells.py [--write]   (without --write: exit 1 when the committed tables differ from the generated ones)
"""
import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HEADER = HERE.parent.parent / "tt_llk_blackhole" / "llk_lib" / "llk_matmul_forms.h"
SWEEP = HERE / "helpers" / "matmul_sweep.py"
PROD = HERE / "helpers" / "matmul_prod_keys.py"
SHORT = {
    "Float16_b": "F16b",
    "Bfp8_b": "B8",
    "Bfp4_b": "B4",
    "Float32": "F32",
    "Float16": "F16",
    "Bfp8": "B8x",
    "Bfp2_b": "B2",
    "Int8": "I8",
    "UInt8": "U8",
    "Int32": "I32",
}


def cells():
    rows = list(csv.DictReader(open(HERE / "matmul_closed_off_rows.csv")))
    keys = {}
    for r in rows:
        k = (
            r["fidelity"],
            r["in_a"],
            r["in_b"],
            r["out"],
            int(r["rt"]),
            int(r["ct"]),
            int(r["kt"]),
            int(r["throttle"]),
            r["l1"] == "True",
            r.get("sync") == "Full",
        )
        keys.setdefault(k, []).append(r)
    per_tile, main = [], []
    for k, rs in sorted(keys.items()):
        slow = [r for r in rs if r["closed_slow"] == "True"]
        if not slow:
            continue
        if any(r.get("form") for r in slow):
            # the measured fastest form that is not slower than main, per row (pt: per-tile pack for the key, main: main's whole form for the row)
            if any(r["form"] == "pt" for r in slow):
                per_tile.append(k)
            for acc, nb in sorted(
                {
                    (r["acc"] == "True", int(r["nb"]))
                    for r in slow
                    if r["form"] == "main"
                }
            ):
                main.append((k, acc, nb))
        elif any(r["pt_slow"] == "True" for r in rs):
            for acc, nb in sorted({(r["acc"] == "True", int(r["nb"])) for r in slow}):
                main.append((k, acc, nb))
        else:
            per_tile.append(k)
    return per_tile, main


def reinit_cells():
    """Cells that keep their keyed form under init before every call (csv column reinit = keyed); every other cell re-inits on main's form."""
    keyed = set()
    for r in csv.DictReader(open(HERE / "matmul_closed_off_rows.csv")):
        if r.get("reinit") != "keyed":
            continue
        keyed.add(
            (
                r["fidelity"],
                r["in_a"],
                r["in_b"],
                r["out"],
                int(r["rt"]),
                int(r["ct"]),
                int(r["kt"]),
                int(r["throttle"]),
                r["l1"] == "True",
                r.get("sync") == "Full",
                r["acc"] == "True",
                int(r["nb"]),
            )
        )
    return sorted(keyed)


def reinit_block():
    return "".join(
        '    ("%s", %s, %s, "%s", "%s", "%s", %d, %d, %d, %d, %s, %d),\n'
        % (
            k[0],
            "False" if k[9] else "True",
            k[10],
            SHORT[k[1]],
            SHORT[k[2]],
            SHORT[k[3]],
            k[4],
            k[5],
            k[6],
            k[7],
            k[8],
            k[11],
        )
        for k in reinit_cells()
    )


def blocks():
    per_tile, main = cells()
    cpp_pt = "".join(
        "    {MathFidelity::%s, %d, DataFormat::%s, DataFormat::%s, DataFormat::%s, %d, %d, %d, %s},\n"
        % (k[0], k[7], k[1], k[2], k[3], k[4], k[5], k[6], "true" if k[8] else "false")
        for k in per_tile
    )
    cpp_main = "".join(
        "    {MathFidelity::%s, %s, %s, DataFormat::%s, DataFormat::%s, DataFormat::%s, %d, %d, %d, %d, %s, %d},\n"
        % (
            k[0],
            "false" if k[9] else "true",
            "true" if acc else "false",
            k[1],
            k[2],
            k[3],
            k[4],
            k[5],
            k[6],
            k[7],
            "true" if k[8] else "false",
            nb,
        )
        for k, acc, nb in main
    )
    py_pt = "".join(
        '    ("%s", %d, "%s", "%s", "%s", %d, %d, %d, %s),\n'
        % (k[0], k[7], SHORT[k[1]], SHORT[k[2]], SHORT[k[3]], k[4], k[5], k[6], k[8])
        for k in per_tile
    )
    py_main = "".join(
        '    ("%s", %s, %s, "%s", "%s", "%s", %d, %d, %d, %d, %s, %d),\n'
        % (
            k[0],
            "False" if k[9] else "True",
            acc,
            SHORT[k[1]],
            SHORT[k[2]],
            SHORT[k[3]],
            k[4],
            k[5],
            k[6],
            k[7],
            k[8],
            nb,
        )
        for k, acc, nb in main
    )
    return cpp_pt, cpp_main, py_pt, py_main


def patch(path, tag, body, comment):
    s = path.read_text()
    pat = re.compile(
        r"(%s BEGIN generated by gen_matmul_closed_off_cells.py[^\n]*\n)(.*?)([ \t]*%s END generated\n)"
        % (re.escape(comment), re.escape(comment)),
        re.S,
    )
    m = [x for x in pat.finditer(s) if x.group(1).strip().endswith(" " + tag)]
    assert len(m) == 1, (path, tag)
    x = m[0]
    return (
        s[: x.start(2)] + body + s[x.end(2) :].lstrip("\n")
        if False
        else s[: x.start(2)] + body + s[x.start(3) :]
    ), x.group(2) != body


def main():
    cpp_pt, cpp_main, py_pt, py_main = blocks()
    changed = False
    for path, tag, body, comment in (
        (HEADER, "closed_off", cpp_pt, "//"),
        (HEADER, "grid_main", cpp_main, "//"),
        (SWEEP, "closed_off", py_pt, "#"),
        (PROD, "grid_main", py_main, "#"),
        (PROD, "reinit_keyed", reinit_block(), "#"),
    ):
        text, diff = patch(path, tag, body, comment)
        changed |= diff
        if "--write" in sys.argv and diff:
            path.write_text(text)
    n_pt, n_main = body_counts()
    print(
        f"{n_pt} per-tile cells, {n_main} main-form cells, {len(reinit_cells())} re-init keyed cells;",
        "tables differ" if changed else "tables match",
    )
    return 1 if (changed and "--write" not in sys.argv) else 0


def body_counts():
    pt, m = cells()
    return len(pt), len(m)


if __name__ == "__main__":
    sys.exit(main())

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summary table of run_galaxy_overlap.sh's logs (or any test_flat_combine_overlap.py perf logs).

  python summarize_overlap.py serial=<log> rows=<log> xdown=<log> [acc_rows=<log> acc_xdown=<log>]

serial: FLAT_CMB_NO_OVERLAP=1 (flat on its own whole-grid plan, then combine). Each overlap log also times its own
flat (the overlap's plan, alone) then combine. "best serial" is the fastest of those back-to-back sums; "saved" is the
overlap against it. Accuracy logs: pytest's verdict per case (overlap bit-identical to flat then combine, PCC vs the
PyTorch reference)."""

import re
import sys

OV = re.compile(
    r"flat\+combine (\S+) (\S+) traced: sequential ([\d.]+) us \(flat ([\d.]+) \+ combine ([\d.]+)\), overlap ([\d.]+) us"
)
SER = re.compile(r"flat\+combine (\S+) (\S+) traced: flat ([\d.]+) \+ combine ([\d.]+) = ([\d.]+) us")
ACC = re.compile(r"(PASSED|FAILED|ERROR)\S*\s.*?test_flat_combine_overlap\[([^\]]+)\]|test_flat_combine_overlap\[([^\]]+)\]\S* (PASSED|FAILED|ERROR)")


def parse(path, rx):
    try:
        return rx.findall(open(path, errors="replace").read())
    except FileNotFoundError:
        return []


def main():
    logs = dict(a.split("=", 1) for a in sys.argv[1:])
    serial = {(m, t): (float(f), float(c), float(s)) for m, t, f, c, s in parse(logs.get("serial", ""), SER)}
    ov = {}
    for lay in ("rows", "xdown"):
        for m, t, s, f, c, o in parse(logs.get(lay, ""), OV):
            ov[(lay, m, t)] = (float(s), float(f), float(c), float(o))
    cases = sorted({k for k in serial} | {(m, t) for _, m, t in ov})
    out = [
        "| model | routing | serial: flat (whole grid) + combine | rows 1-9: flat / overlap | + row-0 cells: flat / "
        "overlap | best serial | best overlap | saved | combine hidden |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m, t in cases:
        seqs, ovs, comb = [], [], None
        cell = {}
        if (m, t) in serial:
            f, c, s = serial[(m, t)]
            seqs.append(s)
            comb = c
            cell["serial"] = f"{f:.1f} + {c:.1f} = {s:.1f}"
        for lay in ("rows", "xdown"):
            if (lay, m, t) in ov:
                s, f, c, o = ov[(lay, m, t)]
                seqs.append(s)
                ovs.append((o, lay))
                comb = comb or c
                cell[lay] = f"{f:.1f} / {o:.1f}"
        best_s = min(seqs) if seqs else None
        best_o = min(ovs) if ovs else None
        if best_s and best_o:
            saved = best_s - best_o[0]
            hidden = f"{min(max(saved, 0), comb) / comb:.0%}" if comb else "-"
            tail = f"{best_s:.1f} | {best_o[0]:.1f} ({best_o[1]}) | {saved / best_s:+.1%} | {hidden}"
        else:
            tail = "- | - | - | -"
        out.append(f"| {m} | {t} | {cell.get('serial', '-')} | {cell.get('rows', '-')} | {cell.get('xdown', '-')} | {tail} |")
    for lay in ("acc_rows", "acc_xdown"):
        if lay in logs:
            res = {b or c: a or d for a, b, c, d in parse(logs[lay], ACC)}  # (inline and -rA lines: once per case)
            out.append(f"\n{lay}: " + (", ".join(f"{case} {v}" for case, v in res.items()) or "no results (see the log)"))
    print("\n".join(out))


if __name__ == "__main__":
    main()

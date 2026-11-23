#!/usr/bin/env python3
"""Compare static counts (per build averages, -k add, 10 builds per side) of the asserts-on rebuild (control),
the asserts-off rebuild and the run's own postcheck tt_tti_counts.md. Prints/writes counts only (no code).
Usage: compare.py <out dir> <hal-runs dir> [--md F] [--json F] [--runs name:dir,name:dir,...]
--runs (added 2026-11-22 for 8a-8e) replaces the built-in 7a-8a list; dir = run dir (absolute, or relative to <hal-runs dir>).
Without --runs the behaviour is unchanged."""
import argparse
import json
import os
import statistics
import sys

RUNS = {"7a": "20261104_211905_B_c0e38f1f_claude-opus-5-5", "7b": "20261105_102435_B_c0e38f1f_claude-opus-5-5",
        "7c": "20261105_122703_B_c0e38f1f_claude-opus-5-5", "7d": "20261106_020508_B_c0e38f1f_claude-opus-5-5",
        "7e": "20261106_095649_B_c0e38f1f_claude-opus-5-5", "8a": "20261119_163437_B_c0e38f1f_claude-opus-5-5"}
COLS = {"text bytes": "text", "TTI": "tti", "TT elfray": "tt_elfray", "TT linear": "tt_linear",
        "TT all paths": "tt_all", "unresolved": "unres", "TTI + TT all paths": "tti_tt"}


def parse_avg(path):
    """'### Average per build' table of tt_tti_counts.md -> {(side, kernel): {col: value}}."""
    if not os.path.isfile(path):
        return None
    lines = open(path).read().splitlines()
    try:
        i = next(k for k, l in enumerate(lines) if l.strip().startswith("### Average per build"))
    except StopIteration:
        return None
    rows, header = {}, None
    for l in lines[i + 1:]:
        if not l.startswith("|"):
            if header:
                break
            continue
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if set(cells[0]) <= {"-"}:
            continue
        d = {}
        for h, c in zip(header[2:], cells[2:]):
            if h in COLS:
                d[COLS[h]] = float(c)
        rows[(cells[0], cells[1])] = d
    return rows


def codesize_text(path):
    """codesize.json -> {'orig': sum of per-kernel mean text, 'hal': ...} (riscv size measure, secondary)."""
    if not os.path.isfile(path):
        return None
    pc = json.load(open(path)).get("per_component", {})
    out = {}
    for side in ("orig", "hal"):
        try:
            out[side] = round(sum(pc[k][side]["text"]["mean"] for k in ("unpack", "math", "pack")), 1)
        except KeyError:
            out[side] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("runs_root")
    ap.add_argument("--md")
    ap.add_argument("--json")
    ap.add_argument("--runs")
    a = ap.parse_args()
    runs = dict(x.split(":", 1) for x in a.runs.split(",")) if a.runs else RUNS
    res, problems = {}, []
    for n, d in runs.items():
        pc = parse_avg(os.path.join(a.runs_root, d, f"postcheck_{n}", "tt_tti_counts.md"))
        on = parse_avg(os.path.join(a.out, n, "assert_on", "tt_tti_counts.md"))
        off = parse_avg(os.path.join(a.out, n, "assert_off", "tt_tti_counts.md"))
        cs_on = codesize_text(os.path.join(a.out, n, "assert_on", "codesize.json"))
        cs_off = codesize_text(os.path.join(a.out, n, "assert_off", "codesize.json"))
        if not (pc and on and off):
            problems.append(f"{n}: missing table (postcheck {bool(pc)}, on {bool(on)}, off {bool(off)})")
            continue
        r = {}
        for side in ("original", "hal"):
            k = (side, "sum")
            r[side] = {"postcheck": pc.get(k), "assert_on": on.get(k), "assert_off": off.get(k),
                       "kernels_off": {kk: off.get((side, kk)) for kk in ("unpack", "math", "pack")}}
            if pc.get(k) != on.get(k):
                problems.append(f"{n} {side}: CONTROL MISMATCH postcheck {pc.get(k)} vs assert_on {on.get(k)}")
        r["codesize_text"] = {"assert_on": cs_on, "assert_off": cs_off}
        res[n] = r
    origs = {json.dumps(v["original"]["assert_off"], sort_keys=True) for v in res.values()}
    if len(origs) > 1:
        problems.append(f"original asserts-off counts differ between runs ({len(origs)} distinct)")
    hal7 = [res[n]["hal"]["assert_off"] for n in ("7a", "7b", "7c", "7d", "7e") if n in res]
    med = {c: statistics.median(h[c] for h in hal7) for c in ("text", "tti", "tt_all", "tti_tt")} if len(hal7) == 5 else None

    L = ["# Asserts on/off rebuild: static counts per build (test_eltwise_binary -k add, 10 builds per side)", "",
         "Source: tools/asserts_off/out/<run>/{assert_on,assert_off}/tt_tti_counts.md and the run's postcheck_<run>/tt_tti_counts.md.",
         "Control: assert_on must equal postcheck (same build as the harness code-size step).", "",
         "| Run | Side | .text postcheck | .text on | .text off | TTI on | TTI off | TT all on | TT all off | TTI+TT on | TTI+TT off | unresolved off | control |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in res.items():
        for side in ("original", "hal"):
            p, on, off = r[side]["postcheck"], r[side]["assert_on"], r[side]["assert_off"]
            ok = "OK" if p == on else "MISMATCH"
            L.append(f"| {n} | {side} | {p['text']} | {on['text']} | {off['text']} | {on['tti']} | {off['tti']} | "
                     f"{on['tt_all']} | {off['tt_all']} | {on['tti_tt']} | {off['tti_tt']} | {off['unres']} | {ok} |")
    if med:
        L += ["", f"HAL 7a-7e asserts off, median per build: .text {med['text']}, TTI {med['tti']}, "
              f"TT all paths {med['tt_all']}, TTI+TT {med['tti_tt']}"]
    L += ["", "Secondary measure (codesize.py, sum of per-kernel mean `text` from riscv size):", "",
          "| Run | orig on | orig off | HAL on | HAL off |", "|---|---|---|---|---|"]
    for n, r in res.items():
        c_on, c_off = r["codesize_text"]["assert_on"] or {}, r["codesize_text"]["assert_off"] or {}
        L.append(f"| {n} | {c_on.get('orig')} | {c_off.get('orig')} | {c_on.get('hal')} | {c_off.get('hal')} |")
    L += ["", "Problems: " + ("none" if not problems else "; ".join(problems))]
    text = "\n".join(L) + "\n"
    print(text)
    if a.md:
        open(a.md, "w").write(text)
    if a.json:
        json.dump({"runs": res, "hal_7a_7e_off_median": med, "problems": problems}, open(a.json, "w"), indent=1)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

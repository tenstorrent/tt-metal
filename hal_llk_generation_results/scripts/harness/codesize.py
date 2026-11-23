#!/usr/bin/env python3
"""
Code size and instruction statistics: original kernels versus HAL kernels.

Reads the ELFs that the tt-llk test framework leaves in
    <artefacts>/sources/<test>.cpp/<variant_id>/elf/{unpack,math,pack}.elf
and computes for each of them:
  - section sizes (text / data / bss) via riscv-tt-elf-size
  - the number of instructions from the disassembly (riscv-tt-elf-objdump -d), split into
    Tensix instructions and ordinary RISC-V instructions, plus a mnemonic histogram.

Variants are paired by variant_id (same parameters -> same id) when possible.

Usage:
  codesize.py --artefacts <dir/tt-llk-build> --tools <tests/sfpi/compiler/bin>
              [--orig eltwise_binary_test.cpp] [--hal eltwise_binary_hal_test.cpp]
              [--sample N] [--json out.json] [--md out.md] [--csv out.csv]

Recognizing Tensix instructions:
  By default an instruction counts as a Tensix instruction if its mnemonic matches the regex
  TENSIX_RE (env), default '^(tt[a-z0-9_]*|\\.insn|\\.word)$'. Check on one
  disassembly (option --dump-one) how your objdump shows Tensix instructions
  and adjust TENSIX_RE if needed. Also: TTI_* instructions are embedded directly
  in the RISC-V code, while TT_* instructions are created at runtime (a write to the instruction buffer),
  so TT_* do not show up as a Tensix mnemonic but as a store; therefore the TT/TTI ratio here is
  an estimate and should be confirmed with the elfray tool when it becomes available.
"""
import argparse
import collections
import csv
import json
import os
import random
import re
import statistics
import subprocess
from pathlib import Path

COMPONENTS = ("unpack", "math", "pack")
TENSIX_RE = re.compile(os.environ.get("TENSIX_RE", r"^(tt[a-z0-9_]*|\.insn|\.word)$"))
ADDR_RE = re.compile(r"^\s*[0-9a-f]+:$")


def find_variants(artefacts: Path, test: str):
    base = artefacts / "sources" / test
    if not base.is_dir():
        cands = [p for p in (artefacts / "sources").glob(f"{test}*") if p.is_dir()] if (artefacts / "sources").is_dir() else []
        base = cands[0] if cands else base
    out = {}
    if base.is_dir():
        for v in base.iterdir():
            elf = v / "elf"
            if all((elf / f"{c}.elf").exists() for c in COMPONENTS):
                out[v.name] = elf
    return out


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout


def sizes(tools: Path, elf: Path):
    out = run([str(tools / "riscv-tt-elf-size"), str(elf)]).splitlines()
    if len(out) >= 2:
        t, d, b = (int(x) for x in out[1].split()[:3])
        return {"text": t, "data": d, "bss": b}
    return {"text": None, "data": None, "bss": None}


def disasm(tools: Path, elf: Path):
    hist = collections.Counter()
    for line in run([str(tools / "riscv-tt-elf-objdump"), "-d", str(elf)]).splitlines():
        # objdump format: "  addr:<TAB>bytes<TAB>mnemonic<TAB>operands"
        parts = line.split("\t")
        if len(parts) >= 3 and ADDR_RE.match(parts[0]) and parts[2].strip():
            hist[parts[2].split()[0]] += 1
    tensix = sum(n for k, n in hist.items() if TENSIX_RE.match(k))
    return {"instr_total": sum(hist.values()), "instr_tensix": tensix,
            "instr_riscv": sum(hist.values()) - tensix}, hist


def measure(tools, elfdir):
    res, hists = {}, {}
    for c in COMPONENTS:
        e = elfdir / f"{c}.elf"
        d, h = disasm(tools, e)
        res[c] = {**sizes(tools, e), **d}
        hists[c] = h
    return res, hists


def agg(values):
    v = [x for x in values if isinstance(x, (int, float))]
    if not v:
        return None
    return {"mean": round(statistics.mean(v), 1), "median": statistics.median(v), "min": min(v), "max": max(v), "n": len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artefacts", required=True)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--orig", default="eltwise_binary_test.cpp")
    ap.add_argument("--hal", default="eltwise_binary_hal_test.cpp")
    ap.add_argument("--sample", type=int, default=0, help="measure only N random paired variants (0 = all)")
    ap.add_argument("--json"); ap.add_argument("--md"); ap.add_argument("--csv")
    ap.add_argument("--dump-one", action="store_true", help="print the disassembly of one HAL math kernel and exit")
    a = ap.parse_args()
    arte, tools = Path(a.artefacts), Path(a.tools)

    orig, hal = find_variants(arte, a.orig), find_variants(arte, a.hal)
    if a.dump_one:
        any_hal = next(iter(hal.values()), None) or next(iter(orig.values()), None)
        if not any_hal:
            raise SystemExit("no ELFs")
        print(run([str(tools / "riscv-tt-elf-objdump"), "-d", str(any_hal / "math.elf")])[:20000])
        return

    paired = sorted(set(orig) & set(hal))
    if paired:
        keys = paired
        mode = "paired by variant_id"
    else:
        keys = None
        mode = "not paired (different variant_id) - the aggregate of all variants is compared"
    random.seed(0)

    rows = []
    hist_tot = {"orig": collections.Counter(), "hal": collections.Counter()}

    def take(d):
        ks = sorted(d)
        if a.sample and len(ks) > a.sample:
            ks = random.sample(ks, a.sample)
        return ks

    if keys is not None:
        if a.sample and len(keys) > a.sample:
            keys = random.sample(keys, a.sample)
        sel = {"orig": keys, "hal": keys}
    else:
        sel = {"orig": take(orig), "hal": take(hal)}

    for side, src in (("orig", orig), ("hal", hal)):
        for vid in sel[side]:
            m, h = measure(tools, src[vid])
            for c in COMPONENTS:
                rows.append({"side": side, "variant_id": vid, "component": c, **m[c]})
                hist_tot[side].update(h[c])

    summary = {"mode": mode, "variants_orig": len(orig), "variants_hal": len(hal),
               "measured_orig": len(sel["orig"]), "measured_hal": len(sel["hal"]),
               "tensix_re": TENSIX_RE.pattern, "per_component": {}}
    for c in COMPONENTS:
        summary["per_component"][c] = {
            side: {k: agg([r[k] for r in rows if r["side"] == side and r["component"] == c])
                   for k in ("text", "instr_total", "instr_tensix", "instr_riscv")}
            for side in ("orig", "hal")}

    # mnemonic difference (total, normalized by the number of kernels)
    no, nh = max(len(sel["orig"]), 1), max(len(sel["hal"]), 1)
    mn = set(hist_tot["orig"]) | set(hist_tot["hal"])
    diff = sorted(((k, hist_tot["orig"][k] / no, hist_tot["hal"][k] / nh) for k in mn),
                  key=lambda x: abs(x[2] - x[1]), reverse=True)
    summary["mnemonic_diff_top"] = [{"mnemonic": k, "orig_per_variant": round(o, 1), "hal_per_variant": round(h, 1)}
                                    for k, o, h in diff[:30]]

    # markdown
    def f(x, k="mean"):
        return "—" if not x else f'{x[k]}'
    md = [f"Mode: {mode}. Variants measured: original {summary['measured_orig']} / {summary['variants_orig']}, "
          f"HAL {summary['measured_hal']} / {summary['variants_hal']}.", "",
          "| Kernel | Metric (mean per variant) | Original | HAL | Difference |", "|---|---|---|---|---|"]
    for c in COMPONENTS:
        for k, label in (("text", "text section (B)"), ("instr_total", "instructions total"),
                         ("instr_tensix", "Tensix instructions"), ("instr_riscv", "RISC-V instructions")):
            o = summary["per_component"][c]["orig"][k]; h = summary["per_component"][c]["hal"][k]
            d = f"{round(h['mean'] - o['mean'], 1):+}" if o and h else "—"
            md.append(f"| {c} | {label} | {f(o)} | {f(h)} | {d} |")
    md += ["", "Largest differences in mnemonics (average count per kernel triple):", "",
           "| Mnemonic | Original | HAL |", "|---|---|---|"]
    md += [f"| `{d['mnemonic']}` | {d['orig_per_variant']} | {d['hal_per_variant']} |" for d in summary["mnemonic_diff_top"][:20]]
    md += ["", f"_Tensix instruction = mnemonic matching `{TENSIX_RE.pattern}`. "
               "The TT/TTI ratio is an estimate; confirm with the elfray tool._"]
    text = "\n".join(md)
    print(text)
    if a.md: Path(a.md).write_text(text + "\n")
    if a.json: Path(a.json).write_text(json.dumps(summary, indent=2))
    if a.csv and rows:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()

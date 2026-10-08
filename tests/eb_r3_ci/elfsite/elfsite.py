#!/usr/bin/env python3
"""Attribute the differences between two builds of one kernel thread to source call sites. Every instruction of
`objdump -d -l --inlines` is assigned to its innermost inline frame in a file matching SITE_RE (the deepseek_v3_b1 unified
kernels, the fused kernel or a twin), else to its function symbol; per site the Tensix instructions (mnemonic tt*) with
operands, and the instruction count, are compared.
usage: elfsite.py <elf A> <elf B> [site regex] [--shift=N (kernel-file lines of B are N later)]   (library: sites(elf) -> {site: (tt list, count)})"""
import collections
import re
import subprocess
import sys

import os
import shutil

OBJDUMP = next(
    (p for p in (os.environ.get("EB_OBJDUMP"), shutil.which("riscv-tt-elf-objdump"),
                 "/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-objdump", "/work/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump",
                 "/localdev/mvlahovic/llk_analysis_builds/round3/eltwise_binary/wheel/sfpi785/sfpi/compiler/bin/riscv-tt-elf-objdump")
     if p and os.path.exists(p)),
    None,
)
SITE_RE = r"(unified_kernels/[^/:]+\.hpp|fused_ops/.*kernel\.cpp|twins/kernels/[^/:]+\.cpp|kernels/eb_dump_reuse\.cpp):\d+"
INSN = re.compile(r"^\s+[0-9a-f]+:\s+[0-9a-f]{8}\s+(\S+)\s*(.*)$")
LOC = re.compile(r"^(?:inlined by )?(\S+?):(\d+)")


def sites(elf, site_re=SITE_RE, shift=0):
    out = subprocess.run([OBJDUMP, "-d", "-l", "--inlines", elf], capture_output=True, text=True).stdout.splitlines()
    rx = re.compile(site_re)
    chain, func = [], None
    res = collections.defaultdict(lambda: [[], 0])
    for line in out:
        m = INSN.match(line)
        if m:
            site = next((rx.search(c).group(0) for c in chain if rx.search(c)), None) or f"fn:{func}"
            if shift and re.search(r"(fused_ops|twins/kernels)/", site):
                f, _, ln = site.rpartition(":")
                site = f"{f}:{int(ln) - shift}"
            mn, ops = m.group(1), m.group(2)
            res[site][1] += 1
            if mn.startswith("tt"):
                res[site][0].append(f"{mn} {ops.split('#')[0].strip()}")
            chain_done = True
            continue
        if line.startswith("/") or line.startswith("inlined by"):
            if line.startswith("/"):
                chain = [line]
            else:
                chain.append(line)
            continue
        fm = re.match(r"^([A-Za-z_][\w.$]*)\(\):$", line)
        if fm:
            func = fm.group(1)
            continue
        if re.match(r"^[0-9a-f]+ <(.+)>:$", line):
            chain = []
    return {k: (v[0], v[1]) for k, v in res.items()}


def compare(a, b, site_re=SITE_RE, shift=0):
    sa, sb = sites(a, site_re), sites(b, site_re, shift)
    rows = []
    for s in sorted(set(sa) | set(sb)):
        ta, ca = sa.get(s, ([], 0))
        tb, cb = sb.get(s, ([], 0))
        if collections.Counter(ta) != collections.Counter(tb) or ca != cb:
            rows.append((s, ca, cb, collections.Counter(ta), collections.Counter(tb)))
    return rows


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--shift=")]
    shift = next((int(x.split("=")[1]) for x in sys.argv[1:] if x.startswith("--shift=")), 0)
    rx = args[2] if len(args) > 2 else SITE_RE
    for s, ca, cb, ta, tb in compare(args[0], args[1], rx, shift):
        ttdiff = ", ".join(f"{k.split()[0]} {tb[k] - ta[k]:+d}" for k in sorted(set(ta) | set(tb)) if tb[k] != ta[k])
        print(f"{s}: insns {ca} -> {cb}; tensix {sum(ta.values())} -> {sum(tb.values())}" + (f" [{ttdiff[:300]}]" if ttdiff else ""))


if __name__ == "__main__":
    main()

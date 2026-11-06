#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Router bring-up tools (temporary: drop before merge; see README.md).

line_compare.py [--function SUBSTR] [--inline NAME ...] [--top N] BASE.elf NEW.elf: one router function's instructions
per source line, base against new, from addr2line's inline chains.

Compare builds by source line, not by function-name buckets: the router's loop inlines its step functions, so a
function's symbol size says little. Per innermost frame (function, file:line) of every instruction of the function:
the count in each image and the change, largest first; loads and stores (sp-relative ones apart, which are spills or
stack locals). --inline NAME lists the new image's instructions whose inline chain passes through NAME (a helper you
added or changed). --function picks the function by a substring of its demangled name; the default is the router
loop the image has (sender::run, receiver::run, single::sender_pass, single::receiver_pass, in that order).
Counts are removed or added work in instructions, not cycles: a stalled load shows at its first use, and moving code
moves the stall.
"""
import argparse
import collections
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router_tools_common import toolchain_prefix  # noqa: E402

B = toolchain_prefix()
LOOPS = (r"sender::run[<(]", r"receiver::run[<(]", r"single::sender_pass[<(]", r"single::receiver_pass[<(]")
MEM = ("lw", "lh", "lhu", "lb", "lbu", "sw", "sh", "sb")


def sh(args, inp=None):
    return subprocess.run(args, capture_output=True, text=True, input=inp).stdout


def find_function(elf, want):
    syms = []
    for l in sh([B + "nm", "-C", "-S", elf]).splitlines():
        p = l.split(None, 3)
        if len(p) == 4 and p[2] in "tTwW":
            syms.append((int(p[0], 16), int(p[1], 16), p[3]))
    pats = [re.escape(want)] if want else LOOPS
    for pat in pats:
        hit = [s for s in syms if re.search(pat, s[2])]
        if hit:
            return max(hit, key=lambda s: s[1])
    sys.exit(f"{elf}: no function matching {pats}")


def insns(elf, want):
    addr, size, name = find_function(elf, want)
    dis = sh(
        [B + "objdump", "-d", "--no-show-raw-insn", f"--start-address={addr}", f"--stop-address={addr + size}", elf]
    )
    ins = [(int(m.group(1), 16), m.group(2)) for m in re.finditer(r"^\s*([0-9a-f]+):\s+(\S.*)$", dis, re.M)]
    out = sh([B + "addr2line", "-i", "-f", "-C", "-a", "-e", elf], "\n".join(hex(a) for a, _ in ins)).splitlines()
    chains, cur, i = {}, None, 0
    while i < len(out):
        if out[i].startswith("0x"):
            cur = int(out[i], 16)
            chains[cur] = []
            i += 1
            continue
        fn = out[i]
        loc = out[i + 1] if i + 1 < len(out) else "?"
        chains[cur].append((re.sub(r"<.*", "", fn.split("(")[0]).split("::")[-1], loc.split("/")[-1].split(" ")[0]))
        i += 2
    return name, ins, chains


def summarize(elf, want, inline):
    name, ins, ch = insns(elf, want)
    per, mem, hit = collections.Counter(), collections.Counter(), []
    for a, t in ins:
        c = ch.get(a) or [("?", "?")]
        per[f"{c[0][0]} {c[0][1]}"] += 1
        op = t.split()[0]
        if op in MEM:
            mem[("sp " if "(sp)" in t else "") + ("load" if op.startswith("l") else "store")] += 1
        for f, _ in c:
            if f in inline:
                hit.append(f"{f} | {c[0][0]} {c[0][1]} | {t}")
                break
    return name, len(ins), per, mem, hit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--function", default=None)
    ap.add_argument("--inline", action="append", default=[])
    ap.add_argument("--top", type=int, default=60)
    a = ap.parse_args()
    na, ca, pa, ma, _ = summarize(a.base, a.function, set(a.inline))
    nb, cb, pb, mb, hb = summarize(a.new, a.function, set(a.inline))
    print(f"base {na}: {ca} instructions; new {nb}: {cb} ({cb - ca:+d})")
    print(f"loads/stores: base {dict(ma)}  new {dict(mb)}")
    if a.inline:
        print(f"new: {len(hb)} instructions under {', '.join(a.inline)}:")
        for k, v in sorted(collections.Counter(hb).items()):
            print(f"   {v:3d}  {k}")
    d = {k: pb.get(k, 0) - pa.get(k, 0) for k in set(pa) | set(pb)}
    rows = sorted(((k, v) for k, v in d.items() if v), key=lambda kv: (-abs(kv[1]), kv[0]))
    print(f"per innermost (function, file:line), changed lines ({len(rows)}), largest first:")
    for k, v in rows[: a.top]:
        print(f"   {v:+4d}  {k}  ({pa.get(k, 0)} -> {pb.get(k, 0)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

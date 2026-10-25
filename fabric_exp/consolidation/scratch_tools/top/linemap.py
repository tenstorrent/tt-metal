#!/usr/bin/env python3
"""Per-source-line instruction attribution of an ERISC ELF (objdump -dl). Usage: linemap.py elf router.cpp
Prints, for router functions of interest, instruction count, sp loads, sp stores, branches, per variant."""
import re, subprocess, sys
from collections import defaultdict
OD = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"

def parse(elf):
    out = subprocess.run([OD, "-dl", "--section=.text", elf], capture_output=True, text=True).stdout
    cur = ("?", 0)
    res = []
    for l in out.splitlines():
        m = re.match(r"^(/\S+):(\d+)", l)
        if m:
            cur = (m.group(1).rsplit("/", 1)[1], int(m.group(2)))
            continue
        m = re.match(r"^\s+([0-9a-f]+):\t\S+\s*\t(\S+)\s*(.*)$", l)
        if m:
            res.append((int(m.group(1), 16), m.group(2), m.group(3), cur))
    return res

def fn_ranges(src):
    lines = open(src).read().split("\n")
    def find(pat, start=0):
        for i in range(start, len(lines)):
            if pat in lines[i]:
                return i + 1
        raise KeyError(pat)
    r = {}
    a = find("FORCE_INLINE bool sender_channel_try_send(")
    b = find("phase_profiler.close<PHASE_TX_SEND>(send_start);", a)
    r["try_send"] = (a, b + 3)
    a = find("    run_worker_sender_step(")
    b = find("    return progress;", a)
    r["worker_step"] = (a, b)
    a = find("    run_forwarding_sender_step(")
    b = find("    return progress;", a)
    r["fwd_step"] = (a, b)
    return r

if __name__ == "__main__":
    elf, src = sys.argv[1], sys.argv[2]
    ins = parse(elf)
    rng = fn_ranges(src)
    for name, (a, b) in rng.items():
        sel = [x for x in ins if x[3][0] == "fabric_erisc_router_v2.cpp" and a <= x[3][1] <= b]
        ld = sum(1 for x in sel if x[1] == "lw" and "(sp)" in x[2])
        st = sum(1 for x in sel if x[1] == "sw" and "(sp)" in x[2])
        br = sum(1 for x in sel if x[1].startswith("b") or x[1] in ("j", "jal"))
        print(f"{name:12s} insns {len(sel):4d}  sp-loads {ld:3d}  sp-stores {st:3d}  branches/jumps {br:3d}")
    ch = [x for x in ins if x[3][0] == "fabric_erisc_datamover_channels.hpp"]
    print(f"{'channels.hpp':12s} insns {len(ch):4d}")

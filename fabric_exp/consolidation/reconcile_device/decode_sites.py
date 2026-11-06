#!/usr/bin/env python3
"""decode_sites.py ELF... : the instructions objdump -l attributes to fabric_common.h's decode_action_y_first body
(study 279-281, port 358-362) in each router ELF's receiver code, grouped per contiguous run. Read-only."""
import re, subprocess, sys
OD = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
LINES = {int(a) for a in sys.argv[1].split(",")}
for elf in sys.argv[2:]:
    out = subprocess.run([OD, "-d", "-l", "--no-show-raw-insn", elf], capture_output=True, text=True).stdout
    cur, fn, runs, last = None, None, [], None
    for l in out.splitlines():
        m = re.match(r"^(\S+)\(\):$", l)
        if m:
            fn = m.group(1); continue
        m = re.match(r"^(/\S+):(\d+)", l)
        if m:
            cur = (m.group(1), int(m.group(2))); continue
        m = re.match(r"^\s+([0-9a-f]+):\s+(.*)$", l)
        if m and cur and cur[0].endswith("fabric_common.h") and cur[1] in LINES:
            a = int(m.group(1), 16)
            if last is None or a - last > 8:
                runs.append([])
            runs[-1].append(f"{fn}:{cur[1]}: {m.group(2).strip()}")
            last = a
    parts = elf.split('/')
    name = parts[-3] if parts[-2].endswith("erisc") and not parts[-2][0].isdigit() else parts[-2]
    print(f"== {name} ({len(runs)} sites)")
    for r in runs:
        print("   " + " | ".join(r))

#!/usr/bin/env python3
"""blocks.py ELF KEY[,KEY...]: instructions and stack accesses of receiver::run attributed (any inline frame) to KEYs."""
import re, subprocess, sys
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import pcs_sections as ps
B = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-"
elf, keys = sys.argv[1], sys.argv[2].split(",")
show = len(sys.argv) > 3
for l in subprocess.run([B + "nm", "-S", elf], capture_output=True, text=True).stdout.splitlines():
    p = l.split()
    if len(p) == 4 and "receiver3run" in p[3]:
        s, sz = int(p[0], 16), int(p[1], 16)
d = subprocess.run([B + "objdump", "-d", "--no-show-raw-insn", f"--start-address={s}", f"--stop-address={s+sz}", elf],
                   capture_output=True, text=True).stdout
ins = [(int(m.group(1), 16), m.group(2).strip()) for m in (re.match(r"\s*([0-9a-f]+):\s+(.*)$", l) for l in d.splitlines()) if m]
loc = ps.a2l(elf, [a for a, _ in ins])
blk = [(a, t) for a, t in ins if any(k in " ".join(f for f, _, _ in loc.get(a, [])) for k in keys)]
print(f"loop function {len(ins)} insns; block {len(blk)} insns, {sum(1 for _, t in blk if '(sp)' in t)} stack accesses")
if show:
    for a, t in blk:
        fr = loc.get(a, [])
        print(f"   {a:06x}  {t[:50]:<50} {fr[0][1].split('/')[-1] + ':' + str(fr[0][2]) if fr else ''}")

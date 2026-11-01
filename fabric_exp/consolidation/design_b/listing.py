#!/usr/bin/env python3
"""listing.py ELF [FROM_LINE_FILTER]: the receiver loop function (symbol with "receiver3run" or "receiver_pass"),
every instruction with its innermost source line and the inline chain's function names (short), stack accesses
marked '*'. Optional filter: only instructions whose inline chain mentions one of the comma-separated keys."""
import re, subprocess, sys
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import pcs_sections as ps
B = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-"
elf = sys.argv[1]
keys = sys.argv[2].split(",") if len(sys.argv) > 2 else None
sym = None
for l in subprocess.run([B + "nm", "-S", elf], capture_output=True, text=True).stdout.splitlines():
    p = l.split()
    if len(p) == 4 and ("receiver3run" in p[3] or "receiver_pass" in p[3]):
        sym = (int(p[0], 16), int(p[1], 16))
s, sz = sym
d = subprocess.run([B + "objdump", "-d", "--no-show-raw-insn", f"--start-address={s}", f"--stop-address={s + sz}", elf],
                   capture_output=True, text=True).stdout
ins = [(int(m.group(1), 16), m.group(2).strip()) for m in (re.match(r"\s*([0-9a-f]+):\s+(.*)$", l) for l in d.splitlines()) if m]
loc = ps.a2l(elf, [a for a, _ in ins])


def short(fn):
    fn = re.sub(r"\(.*", "", fn)
    fn = re.sub(r"<.*>", "", fn)
    return fn.split("::")[-1]


print(f"{len(ins)} instructions, {sum(1 for _, t in ins if '(sp)' in t)} stack accesses")
for a, t in ins:
    fr = loc.get(a, [])
    chain = "<".join(short(f) for f, _, _ in fr)
    if keys and not any(k in chain for k in keys):
        continue
    where = f"{fr[0][1]}:{fr[0][2]}" if fr else ""
    t = re.sub(r"\s+", " ", t)
    print(f"{'*' if '(sp)' in t else ' '} {a:05x}  {t[:44]:<44} {where:<28} {chain[:110]}")

import re, subprocess, sys
B = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-"
o = sys.argv[1]
sym = next(l.split()[-1] for l in subprocess.run([B + "nm", o], capture_output=True, text=True).stdout.splitlines() if l.split()[-1].startswith("_Z3run"))
d = subprocess.run([B + "objdump", "-d", "--no-show-raw-insn", "--disassemble=" + sym, o], capture_output=True, text=True).stdout
ins = [(int(m.group(1), 16), m.group(2), m.group(3).split("<")[0].strip()) for m in (re.match(r"\s*([0-9a-f]+):\s+(\S+)\s*(.*)", l) for l in d.splitlines()) if m]
# the group read: the run of loads from the stream registers (0xffb4xxxx); print it with what sits between its loads
base = {}
for i, (a, mn, ops) in enumerate(ins):
    if mn == "lui" and "0xffb4" in ops:
        base[ops.split(",")[0]] = a
idx = [i for i, (a, mn, ops) in enumerate(ins) if mn == "lw" and any(f"({r})" in ops for r in base)]
if idx:
    lo, hi = idx[0], idx[0]
    while hi + 1 < len(ins) and hi - lo < 14 and (hi + 1 in idx or ins[hi + 1][1] in ("lui", "addi", "li")):
        hi += 1
    print(" ; ".join(f"{mn} {ops}" for _, mn, ops in ins[lo:hi + 1]))
